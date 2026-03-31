"""
TDA-Informed Robust Importance Sampling with Defensive Refinement
集成 UCB 搜索与全局防御性分布（Global Defensive Mixture）
"""
import os.path as osp
import pickle
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import gpytorch

from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from scipy.special import logsumexp  # 用于数值稳定的对数求和
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from safebench.util.torch_util import set_seed, set_torch_variable
from safebench.util.run_util import load_config
from safebench.carla_runner_simple import CarlaRunner
from rlsan.src.surrogate.utils import load_surrogate_model, save_surrogate_model
from sklearn.cluster import KMeans
warnings.filterwarnings("ignore")


class GaussianPatchManager:
    """动态注入高斯核补丁，修正权重异常"""
    def __init__(self, max_patches=10, alpha_per_patch=0.02):
        self.patches = []  # List of (mean, cov, alpha)
        self.verified_patches = []  # 经过验证的补丁
        self.max_patches = max_patches
        self.alpha_per_patch = alpha_per_patch

    def detect_anomalies(self, samples, weights, gamma=50.0):
        """检测权重异常点"""
        mean_weight = np.mean(weights)
        threshold = gamma * mean_weight
        anomaly_mask = weights > threshold
        anomaly_indices = np.where(anomaly_mask)[0]
        return anomaly_indices

    def inject_patches(self, anomaly_samples, cov_scale=0.02):
        """在异常点处注入高斯补丁"""
        if len(self.patches) >= self.max_patches:
            print(f"  [Patch] Max patches ({self.max_patches}) reached. Skipping new patches.")
            return

        # 计算局部协方差（基于异常样本的方差）
        if len(anomaly_samples) > 1:
            local_cov = np.cov(anomaly_samples.T) * cov_scale
            if local_cov.ndim == 0:
                local_cov = np.array([[local_cov]])
        else:
            local_cov = np.eye(anomaly_samples.shape[1]) * cov_scale

        # 为每个异常点创建补丁
        for x_star in anomaly_samples:
            if len(self.patches) < self.max_patches:
                self.patches.append({
                    'mean': x_star,
                    'cov': local_cov,
                    'alpha': self.alpha_per_patch,
                    'verified': False
                })

        print(f"  [Patch] Injected {len(anomaly_samples)} patches. Total: {len(self.patches)}")

    def verify_patches(self, runner, failure_threshold=0.3, n_samples_per_patch=10):
        """验证补丁附近是否存在真实失效区域"""
        verified = []
        for patch in self.patches:
            if patch['verified']:
                verified.append(patch)
                continue

            # 在补丁中心附近采样
            patch_mean = patch['mean']
            patch_cov = patch['cov']
            test_samples = np.random.multivariate_normal(patch_mean, patch_cov, n_samples_per_patch)
            test_samples = np.clip(test_samples, -1, 1)  # 确保在边界内

            # 真实仿真
            try:
                sim_results = []
                for sample in test_samples:
                    result, _ = runner.run(sample.reshape(1, -1))
                    # 崩溃相当于最大危险 1.0
                    sim_results.append(result if result is not None else 1.0)

                failure_rate = np.mean(np.array(sim_results) > failure_threshold)

                if failure_rate > 0.3:  # 至少 30% 的样本失效
                    patch['verified'] = True
                    verified.append(patch)
                    print(f"  [Verify] Patch at {patch_mean[:2]} VERIFIED (failure_rate={failure_rate:.1%})")
                else:
                    print(f"  [Verify] Patch at {patch_mean[:2]} REJECTED (failure_rate={failure_rate:.1%})")
            except Exception as e:
                print(f"  [Verify] Verification failed for patch: {e}")

        self.verified_patches = verified
        return verified

    def compute_patch_density(self, samples):
        """计算所有补丁在样本处的密度贡献"""
        if len(self.patches) == 0:
            return np.zeros(len(samples))

        log_densities = []
        for patch in self.patches:
            # 计算高斯密度
            diff = samples - patch['mean']
            try:
                cov_inv = np.linalg.inv(patch['cov'])
                mahal = np.sum(diff @ cov_inv * diff, axis=1)
                log_det = np.linalg.slogdet(patch['cov'])[1]
                log_dens = -0.5 * (mahal + log_det + samples.shape[1] * np.log(2 * np.pi))
                log_densities.append(log_dens)
            except np.linalg.LinAlgError:
                log_densities.append(np.full(len(samples), -np.inf))

        # 混合所有补丁的对数密度
        log_densities = np.array(log_densities)
        return logsumexp(log_densities, axis=0)

    def clear(self):
        """清空补丁（在下次 GMM 更新后调用）"""
        self.patches = []


def estimate_truncation_constant(gmm, dim, n_mc=1000000):
    """
    使用蒙特卡洛积分估算 GMM 在 [-1, 1]^D 盒子内的概率质量 Z
    这是保证重要性采样无偏性的核心步骤
    """
    print(f"Estimating truncation constant Z for final GMM (dim={dim})...")
    # 在全空间采样（不受边界限制）
    samples_raw, _ = gmm.sample(n_mc)
    # 统计有多少比例落在了 [-1, 1] 区间内
    valid_mask = np.all((samples_raw >= -1) & (samples_raw <= 1), axis=1)
    Z = np.mean(valid_mask)
    print(f"Final Calibrated Z = {Z:.4f}")
    return max(Z, 1e-6)


def log_target_distribution(x):
    """
    概率计算工具函数 (全部转移到 Log 域以保证数值稳定性)
    计算目标分布的对数概率密度 log p(x)
    假设原始空间为 [-1, 1]^D 的均匀分布。
    体积 V = 2^D. D=3 -> V=8. p(x)=1/8.
    log(p(x)) = -log(8)
    """
    dim = x.shape[1]
    vol = 2.0 ** dim
    return np.full(x.shape[0], -np.log(vol))


def expand_gmm_with_patches(gmm, verified_patches, n_modes_max=10):
    """
    将验证通过的补丁转化为正式 GMM 分量
    返回新的 GMM（分量数增加）
    """
    if len(verified_patches) == 0:
        return gmm

    n_new_components = min(len(verified_patches), n_modes_max - gmm.n_components)
    if n_new_components <= 0:
        print(f"  [Expand] GMM already at max components ({n_modes_max}). Skipping expansion.")
        return gmm

    print(f"  [Expand] Expanding GMM from {gmm.n_components} to {gmm.n_components + n_new_components} components")

    # 收集所有数据用于重新拟合
    old_X = gmm.means_
    old_weights = gmm.weights_

    # 从验证通过的补丁中提取新分量的初始化参数
    new_means = np.array([p['mean'] for p in verified_patches[:n_new_components]])
    new_covs = np.array([p['cov'] for p in verified_patches[:n_new_components]])
    new_weights = np.ones(n_new_components) / (gmm.n_components + n_new_components)

    # 创建新 GMM（分量数增加）
    new_gmm = GaussianMixture(
        n_components=gmm.n_components + n_new_components,
        covariance_type='full',
        random_state=42,
        n_init=5
    )

    # 初始化新 GMM 的参数
    new_gmm.means_ = np.vstack([gmm.means_, new_means])
    new_gmm.covariances_ = np.vstack([gmm.covariances_, new_covs])
    new_gmm.weights_ = np.hstack([gmm.weights_ * (gmm.n_components / (gmm.n_components + n_new_components)),
                                   new_weights])
    new_gmm.weights_ /= np.sum(new_gmm.weights_)
    new_gmm.precisions_cholesky_ = _compute_precision_cholesky(new_gmm.covariances_, 'full')

    print(f"  [Expand] GMM expanded successfully. New shape: {new_gmm.means_.shape}")
    return new_gmm


def log_q_mix_density_defensive(samples, q_ce, alpha, Z_GMM, patch_manager=None):
    """
    全校准版的防御性混合密度计算 - 支持动态补丁注入
    log_q_mix = log( (1-alpha-sum_alpha_j)*(q_ce/Z) + alpha*Uniform + sum(alpha_j*G_j) )
    """
    # 1. 计算 GMM 校准对数概率
    log_q_ce_norm = q_ce.score_samples(samples) - np.log(Z_GMM)

    # 2. 计算全局均匀分布的对数概率
    dim = samples.shape[1]
    vol = 2.0 ** dim
    log_q_global_norm = np.full(samples.shape[0], -np.log(vol))

    # 3. 计算补丁贡献（如果存在）
    if patch_manager is not None and len(patch_manager.patches) > 0:
        log_patch_dens = patch_manager.compute_patch_density(samples)
        total_patch_alpha = len(patch_manager.patches) * patch_manager.alpha_per_patch
        alpha_gmm = 1 - alpha - total_patch_alpha
        alpha_gmm = np.clip(alpha_gmm, 1e-10, 1 - 1e-10)

        log_prob_gmm = np.log(alpha_gmm) + log_q_ce_norm
        log_prob_def = np.log(alpha) + log_q_global_norm
        log_prob_patch = np.log(total_patch_alpha) + log_patch_dens

        return logsumexp(np.vstack([log_prob_gmm, log_prob_def, log_prob_patch]).T, axis=1)
    else:
        # 无补丁时的原始逻辑
        alpha = np.clip(alpha, 1e-10, 1 - 1e-10)
        log_prob_ce = np.log(1 - alpha) + log_q_ce_norm
        log_prob_def = np.log(alpha) + log_q_global_norm
        return logsumexp(np.vstack([log_prob_ce, log_prob_def]).T, axis=1)


def load_tda_model(file_path):
    """加载上一步保存的带有拓扑先验的GMM模型"""
    print(f"Loading TDA-GMM model from {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        return data['gmm'], data['tda_k']
    else:
        raise ValueError("Pickle file format incorrect. Expected dictionary with 'gmm' key.")


def failure_indicator(X_data, gp_model, gp_likelihood, device="cuda"):
    """
    调用 GP 代理模型进行推理 (GPU加速)
    """
    # 转为 Tensor 并移动到 device
    x_tensor = torch.tensor(X_data, dtype=torch.float32).to(device)

    gp_model.eval(); gp_likelihood.eval()

    # 使用 GPyTorch 进行预测
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = gp_likelihood(gp_model(x_tensor))
        Y_pred = posterior.mean.cpu().numpy()
        sigma2 = posterior.variance.cpu().numpy()

    return Y_pred, np.sqrt(sigma2)


def sample_mixture_defensive(gmm, N_samples, alpha):
    """
    混合采样：(1-alpha) 来自 GMM，alpha 来自全局均匀分布
    关键：严格维护混合比例，不允许用均匀分布补充 GMM 不足
    """
    n_ce = int(N_samples * (1 - alpha))
    n_def = N_samples - n_ce

    # GMM 采样 - 增加尝试次数以确保获得足够样本
    samples_ce = []
    max_attempts = 500
    attempt = 0
    samples_per_attempt = max(n_ce * 10, 5000)

    while len(samples_ce) < n_ce and attempt < max_attempts:
        new_s, _ = gmm.sample(samples_per_attempt)
        mask = np.all((new_s >= -1) & (new_s <= 1), axis=1)
        samples_ce.extend(new_s[mask])
        attempt += 1

    if len(samples_ce) < n_ce:
        # 采样失败：从已有样本中重复采样（保持分布一致性）
        print(f"[Warning] GMM sampling insufficient: {len(samples_ce)}/{n_ce} after {max_attempts} attempts")
        n_need = n_ce - len(samples_ce)
        if len(samples_ce) > 0:
            resample_idx = np.random.choice(len(samples_ce), size=n_need, replace=True)
            samples_ce.extend([samples_ce[i] for i in resample_idx])
        else:
            # 无法采样：抛出错误而非使用均匀分布
            raise RuntimeError(f"[Critical] GMM sampling completely failed: 0/{n_ce} samples obtained after {max_attempts} attempts. Check GMM validity.")

    samples_ce = np.array(samples_ce[:n_ce])
    np.random.shuffle(samples_ce)

    # 全局均匀采样 [-1, 1]
    samples_def = np.random.uniform(-1, 1, size=(n_def, gmm.means_.shape[1]))

    res = np.vstack([samples_ce, samples_def])
    np.random.shuffle(res)
    return res


def call_real_simulation(samples, runner):
    """调用CARLA真实仿真"""
    print(f"Running real simulation for {len(samples)} elite cases...")
    sim_results = []
    for idx in range(len(samples)):
        try:
            result, _ = runner.run(samples[idx].reshape(1, -1))
            # 仿真返回 None 代表极端的故障（如 Carla 崩溃），应该判定为高危险惩罚值 1.0
            sim_results.append(result if result is not None else 1.0)
        except Exception as e:
            print(f"Warning: Simulation failed for sample {idx}: {e}. Using penalty value 1.0")
            sim_results.append(1.0)
    return np.array(sim_results).reshape(-1)


def update_surrogate_model(samples, true_values, gp_model, gp_likelihood, device="cuda"):
    """更新代理模型数据"""
    print(f"Updating global surrogate model with {len(samples)} new cases...")

    # 转换为 GPU Tensor
    new_X = torch.tensor(samples, dtype=torch.float32).to(device)
    new_Y = torch.tensor(true_values, dtype=torch.float32).reshape(-1).to(device)

    old_X = gp_model.train_inputs[0]
    old_Y = gp_model.train_targets

    # 拼接数据
    train_x = torch.cat([old_X, new_X])
    train_y = torch.cat([old_Y, new_Y])

    # 设置新训练数据
    gp_model.set_train_data(inputs=train_x, targets=train_y, strict=False)

    # 保存
    save_path = '/home/hp/SENSE/rlsan/results/s1exp/updated_surrogate_model.pkl'
    save_surrogate_model(gp_model, gp_likelihood, save_path)
    print("Global model updated and saved.")


# ================= 增强型 Active Learning 工具 =================
def calculate_u_function(mu, sigma, threshold=0.3):
    """
    计算 U-Function (Learning Function).
    U(x) = |threshold - mu(x)| / sigma(x)
    选择接近阈值的点（边界区域）进行仿真，这些点最不确定。
    U <= 2.0 代表预测错误的概率低于约 2.3%.
    """
    sigma = np.clip(sigma, 1e-6, None)
    return np.abs(threshold - mu) / sigma


def adaptive_diversity_selection(X_candidates, u_scores, u_threshold=2.0, n_max_sim=200):
    """
    自适应多样性筛选逻辑：
    1. 筛选出 U <= u_threshold 的所有不确定样本。
    2. 如果不确定样本数超过最大预算，通过 K-Means 进行多样性下采样。
    """
    # 筛选不确定样本
    uncertain_mask = (u_scores <= u_threshold)
    uncertain_indices = np.where(uncertain_mask)[0]
    n_uncertain = len(uncertain_indices)

    if n_uncertain == 0:
        return np.array([], dtype=int)

    # 情况 A: 不确定样本在预算内，全部仿真
    if n_uncertain <= n_max_sim:
        print(f" [Adaptive] Found {n_uncertain} uncertain samples. Simulating all.")
        return uncertain_indices

    # 情况 B: 不确定样本过多，执行聚类下采样以保证“不重”
    print(f" [Diversity] {n_uncertain} samples uncertain. Selecting top {n_max_sim} via K-Means.")
    X_uncertain = X_candidates[uncertain_indices]
    u_uncertain = u_scores[uncertain_indices]

    # 执行聚类
    kmeans = KMeans(n_clusters=n_max_sim, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X_uncertain)

    selected_in_uncertain = []
    for i in range(n_max_sim):
        cluster_mask = (cluster_labels == i)
        cluster_u = u_uncertain[cluster_mask]
        cluster_idx_map = np.where(cluster_mask)[0]
        # 在簇内选取最不确定的点 (U 最小)
        best_in_cluster = cluster_idx_map[np.argmin(cluster_u)]
        selected_in_uncertain.append(best_in_cluster)

    return uncertain_indices[np.array(selected_in_uncertain)]


def cross_entropy_refine_robust(q_init, gp_model, gp_likelihood, runner, N=10000, Ne_init=1000, eta = 0.5, T=10, lambda_mix=0.5, device="cuda"):
    """
    集成 UCB 挖掘与防御性协方差膨胀的精炼过程 + 动态补丁注入
    """
    q_ce = q_init
    n_modes = q_init.n_components
    history = []
    failure_threshold = 0.3
    patch_manager = GaussianPatchManager(max_patches=10, alpha_per_patch=0.02)

    for t in range(T):
        # --- Step 1: 混合采样 (GMM + Global Uniform) ---
        X_samples = sample_mixture_defensive(q_ce, N, lambda_mix)

        # --- Step 2: UCB 评估 (解决 FN 问题) ---
        mu, sigma = failure_indicator(X_samples, gp_model, gp_likelihood, device=device)

        # --- Step 3: Active Learning (保持不变) ---
        # 计算全局不确定性评分
        u_scores = calculate_u_function(mu, sigma, threshold=failure_threshold)
        # 确定需要真实仿真的样本 (不固定数量，上限 200)
        # u_threshold=2.0 是 AK-MCS 的经典收敛准则
        real_sim_indices = adaptive_diversity_selection(
            X_samples, u_scores, u_threshold=2.0, n_max_sim=200
        )

        if len(real_sim_indices) > 0:
            # 仅对真正”不确定”的点执行高昂的物理仿真
            Y_real = call_real_simulation(X_samples[real_sim_indices], runner)
            # 更新代理模型，消除认知偏差
            update_surrogate_model(X_samples[real_sim_indices], Y_real, gp_model, gp_likelihood, device=device)
            # 修正当前批次的评估值
            mu[real_sim_indices] = Y_real

            # 用刚刚更新的 GP 对未仿真的数据进行一轮重新预测，保证后续抽精英不产生特征滞后
            not_sim_mask = np.ones(len(X_samples), dtype=bool)
            not_sim_mask[real_sim_indices] = False
            mu_new, _ = failure_indicator(X_samples[not_sim_mask], gp_model, gp_likelihood, device=device)
            mu[not_sim_mask] = mu_new
        else: 
            print(f"  [Skip] Surrogate is sufficiently confident for this iteration.")

        # --- Step 4: 筛选精英并拟合 ---
        # 使用经过真值校准后的均值 mu 进行精英筛选
        # 选择最高的 Ne_init 个值（最危险的场景）
        elite_idx = np.argsort(mu)[-Ne_init:]
        elites = X_samples[elite_idx]
        new_gmm = GaussianMixture(n_components=n_modes, covariance_type='full', random_state=42 + t).fit(elites)

        # 利用匈牙利算法对齐 GMM Component (避免 Label Switching 摧毁拓扑先验)
        cost_matrix = cdist(q_ce.means_, new_gmm.means_, metric='cosine')
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 强制性对齐 new_gmm 的组件序列，使其成分分别吻合于 q_ce 的对应成分
        new_gmm.means_ = new_gmm.means_[col_ind]
        new_gmm.covariances_ = new_gmm.covariances_[col_ind]
        new_gmm.weights_ = new_gmm.weights_[col_ind]

        # --- Step 5: 【核心修改】防御性协方差膨胀 (Cov-Inflation) ---
        # 强制增加 min_eig 以对冲代理模型收缩过快
        min_eig = 0.08
        for i in range(new_gmm.n_components):
            vals, vecs = np.linalg.eigh(new_gmm.covariances_[i])
            vals = np.clip(vals, min_eig, None)
            new_gmm.covariances_[i] = vecs @ np.diag(vals) @ vecs.T
        new_gmm.precisions_cholesky_ = _compute_precision_cholesky(new_gmm.covariances_, 'full')

        # --- Step 6: 动量更新 ---
        q_ce.means_ = (1 - eta) * q_ce.means_ + eta * new_gmm.means_
        q_ce.covariances_ = (1 - eta) * q_ce.covariances_ + eta * new_gmm.covariances_
        q_ce.weights_ = (1 - eta) * q_ce.weights_ + eta * new_gmm.weights_
        q_ce.weights_ /= np.sum(q_ce.weights_)  # 消除浮点误差，保证权重和为1

        # 确保协方差矩阵正定性
        for i in range(q_ce.n_components):
            eigvals = np.linalg.eigvalsh(q_ce.covariances_[i])
            if np.min(eigvals) < 1e-6:
                q_ce.covariances_[i] += np.eye(q_ce.covariances_[i].shape[0]) * (1e-6 - np.min(eigvals))

        q_ce.precisions_cholesky_ = _compute_precision_cholesky(q_ce.covariances_, 'full')

        curr_score = np.mean(mu[elite_idx])
        history.append(curr_score)

        # --- Step 7: 增强型 ESS 监控与混合率动态调整 ---
        # 在 GMM 更新后立即计算截断常数，确保 ESS 计算使用最新的 Z
        t_dim = q_ce.means_.shape[1]
        Z_t = estimate_truncation_constant(q_ce, t_dim)
        X_diag = sample_mixture_defensive(q_ce, N_samples=20000, alpha=lambda_mix)
        log_q_diag = log_q_mix_density_defensive(X_diag, q_ce, lambda_mix, Z_t)
        log_p_diag = log_target_distribution(X_diag)
        log_weights = log_p_diag - log_q_diag
        max_log_w = np.max(log_weights)
        if np.isfinite(max_log_w):
            weights = np.exp(log_weights - max_log_w)
        else:
            weights = np.ones_like(log_weights) / len(log_weights)
        # ESS = (sum(w))^2 / sum(w^2)，归一化为 [0,1]
        ESS_ratio = (np.sum(weights) ** 2) / np.sum(weights ** 2) / len(weights)

        # 权重稳定性反馈机制：检测异常权重
        max_weight = np.max(weights)
        mean_weight = np.mean(weights)
        weight_ratio = max_weight / (mean_weight + 1e-10)

        if weight_ratio > 100:  # 权重爆炸阈值
            print(f"  [WARNING] Weight explosion detected: max/mean ratio = {weight_ratio:.1f}")
            # 触发补丁注入机制
            anomaly_indices = patch_manager.detect_anomalies(X_diag, weights, gamma=50.0)
            if len(anomaly_indices) > 0:
                anomaly_samples = X_diag[anomaly_indices]
                patch_manager.inject_patches(anomaly_samples, cov_scale=0.02)
                # 重新计算权重（基于注入补丁后的新分布）
                log_q_diag_new = log_q_mix_density_defensive(X_diag, q_ce, lambda_mix, Z_t, patch_manager)
                log_weights_new = log_p_diag - log_q_diag_new
                max_log_w_new = np.max(log_weights_new)
                if np.isfinite(max_log_w_new):
                    weights = np.exp(log_weights_new - max_log_w_new)
                else:
                    weights = np.ones_like(log_weights_new) / len(log_weights_new)
                ESS_ratio = (np.sum(weights) ** 2) / np.sum(weights ** 2) / len(weights)
                print(f"  -> Patches injected. New ESS ratio: {ESS_ratio:.2%}")
            # 增加防御分布权重
            lambda_mix = min(0.85, lambda_mix + 0.1)
            print(f"  -> Emergency: increasing lambda to {lambda_mix:.2f} to stabilize weights")
        else:
            # 调整 lambda_mix 动态逻辑：更慢的衰减，保持警觉
            if ESS_ratio < 0.3:
                # 权重太集中，增加由于防御分布的权重
                lambda_mix = min(0.75, lambda_mix + 0.05)
                print(f"  -> ESS Low ({ESS_ratio:.2%}), increasing lambda to {lambda_mix:.2f}")
            elif ESS_ratio > 0.6:
                # 权重非常均匀，尝试非常缓慢地减少
                lambda_mix = max(0.25, lambda_mix - 0.05)
                print(f"  -> ESS High ({ESS_ratio:.2%}), decreasing lambda to {lambda_mix:.2f}")
            else:
                # 保持现状
                print(f"  -> ESS Moderate ({ESS_ratio:.2%}), keeping lambda at {lambda_mix:.2f}")

        print(f"Iter {t + 1}: Elite Mean Score = {curr_score:.4f} | Mix_Rate={lambda_mix:.2f}")

        # 每 3 次迭代检查一次补丁，进行验证和 GMM 扩展
        if (t + 1) % 3 == 0 and len(patch_manager.patches) > 0:
            print(f"\n[Checkpoint] Verifying patches at iteration {t + 1}...")
            verified = patch_manager.verify_patches(runner, failure_threshold=0.3, n_samples_per_patch=10)

            if len(verified) > 0:
                # 将验证通过的补丁转化为正式分量
                q_ce = expand_gmm_with_patches(q_ce, verified, n_modes_max=10)
                n_modes = q_ce.n_components
                print(f"[Checkpoint] GMM expanded to {n_modes} components\n")

            # 清空所有补丁（已验证的转化为分量，未验证的丢弃）
            patch_manager.clear()
        else:
            # 其他迭代只清空补丁
            patch_manager.clear()

    return q_ce, lambda_mix, history


if __name__ == "__main__":
    # 参数解析 (CarlaRunner 需要)
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='rlsan')
    parser.add_argument('--output_dir', type=str, default='log')
    parser.add_argument('--ROOT_DIR', type=str, default='/home/hp/SENSE')
    parser.add_argument('--auto_ego', type=bool, default=False)

    parser.add_argument('--max_episode_step', type=int, default=2000)
    parser.add_argument('--agent_cfg', nargs='+', type=str, default='behavior.yaml')
    parser.add_argument('--scenario_cfg', nargs='+', type=str, default='standard.yaml')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--seed', '-s', type=int, default=42)
    parser.add_argument('--threads', type=int, default=8)

    parser.add_argument('--num_scenario', '-ns', type=int, default=1, help='num of scenarios we run in one episode')
    parser.add_argument('--save_video', type=bool, default=False)
    parser.add_argument('--render', type=bool, default=True)
    parser.add_argument('--frame_skip', '-fs', type=int, default=1, help='skip of frame in each step')
    parser.add_argument('--port', type=int, default=2008, help='port to communicate with carla')
    parser.add_argument('--tm_port', type=int, default=8008, help='traffic manager port')
    parser.add_argument('--fixed_delta_seconds', type=float, default=0.1)

    # 添加其他必要的 Carla 参数...
    args = parser.parse_args()
    args_dict = vars(args)
    set_torch_variable(args.device)
    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    # 1. 加载 Surrogate GP
    gp_path = '../../results/s1exp/surrogate_model_1000.pkl'
    gp_model, gp_likelihood = load_surrogate_model(gp_path)
    gp_model.to(args.device); gp_likelihood.to(args.device)
    print("GP Model loaded.")

    # 加载 TDA-Informed GMM
    tda_model_path = '../../results/s1exp/tda_gmm_model.pkl'
    try:
        q_rl, k_optimal = load_tda_model(tda_model_path)
        print(f"TDA-GMM loaded. Components: {k_optimal}, Means shape: {q_rl.means_.shape}")
    except FileNotFoundError:
        print("Error: TDA model not found. Please run the TDA analysis script first.")
        exit(1)

    # 2. 初始化 CARLA Runner (仅用于 Active Learning)
    agent_config_path = osp.join(args.ROOT_DIR, 'safebench/agent/config', args.agent_cfg)
    agent_config = load_config(agent_config_path)
    agent_config.update(args_dict)

    # load scenario config
    scenario_config_path = osp.join(args.ROOT_DIR, 'rlsan/config', args.scenario_cfg)
    scenario_config = load_config(scenario_config_path)
    scenario_config.update(args_dict)  # 将在本脚本中指定的参数更新到配置中
    runner = CarlaRunner(agent_config, scenario_config, step_by_step=True)

    # 3. 执行 RL-informed Cross Entropy Refinement, 用 TDA 模型作为 q_init
    lambda_mix_init = 0.50  # 初始 mix rate

    final_q_ce, final_lambda_mix, history = cross_entropy_refine_robust(
        q_rl, gp_model, gp_likelihood, runner,
        N=10000, Ne_init=1000, eta=0.5, T=15, lambda_mix=lambda_mix_init, device=args.device
    )

    # 截断校准
    dim = final_q_ce.means_.shape[1]
    final_Zt = estimate_truncation_constant(final_q_ce, dim)

    # 4. 生成最终用于CARLA仿真的评估样本 (IS Evaluation)
    print("Generating samples for final IS estimation...")
    N_total = 20000
    # 使用修改后的均匀混合采样
    X_final = sample_mixture_defensive(final_q_ce, N_total, alpha=final_lambda_mix)
    np.random.shuffle(X_final)

    # 6. 保存最终精炼结果（含完整模型状态，解决版本冲突）
    log_q_mix_values = log_q_mix_density_defensive(X_final, final_q_ce, alpha=final_lambda_mix, Z_GMM=final_Zt, patch_manager=None)
    save_package = {
        'gmm': final_q_ce,
        'lambda_mix': final_lambda_mix,
        'Z_constant': final_Zt,
        'history': history,
        'q_mix_values': np.exp(log_q_mix_values),
        'log_q_mix_values': log_q_mix_values,
     }
    
    # optional: 为了验证代理模型的精确性,另存重要性分布的样本并进行实际仿真
    with open('../../results/s1exp/final_refined_IS_package.pkl', "wb") as f:
        pickle.dump(save_package, f)
    with open('../../results/IS/scenario01/sampled_parameters.pkl', "wb") as f:
        pickle.dump(X_final, f)
    print(f"All data calibrated and saved with Z={final_Zt:.4f}")

    # 8. 绘制历史
    plt.figure()
    plt.plot(history, marker='o')
    plt.title("CE Refinement History")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Elite Prob")
    plt.savefig('../../results/s1exp/ce_history.png')
    plt.close()