"""
TDA-Informed Robust Importance Sampling with Defensive Refinement
集成 UCB 搜索与全局防御性分布（Global Defensive Mixture）
"""
import os.path as osp
import pickle
import argparse
import openpyxl
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

from safebench.util.torch_util import set_seed, set_torch_variable
from safebench.util.run_util import load_config
from safebench.carla_runner_simple import CarlaRunner
from rlsan.src.surrogate.utils import load_surrogate_model, save_surrogate_model
# 忽略特定警告
warnings.filterwarnings("ignore")


def estimate_truncation_constant(gmm, dim, n_mc=1000000):
    """
    使用蒙特卡洛积分估算 GMM 在 [-1, 1]^D 盒子内的概率质量 Z
    这是保证重要性采样无偏性的核心步骤
    """
    print(f"Estimating truncation constant Z for final GMM (dim={dim})...")
    # 在全空间采样（不受边界限制）
    samples_raw, _ = gmm.sample(n_mc)
    # 统计有多少比例落在了 [-1, 1] 盒子内
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


def log_q_mix_density_defensive(samples, q_ce, alpha, Z_GMM, sigma_global=0.8):
    """
    全校准版的防御性混合密度计算
    log_q_mix = log( (1-alpha)*(q_ce/Z) + alpha*Normal(0, sigma_global) )
    """
    # 1. 计算 GMM 校准对数概率
    log_q_ce_norm = q_ce.score_samples(samples) - np.log(Z_GMM)

    # 2. 计算全局高斯的截断常数 Z_global (可预计算或缓存)
    # 对于 D 维独立同分布高斯，Z_global = [erf(1/(sigma*sqrt(2))) - erf(-1/(sigma*sqrt(2)))]^D / 2^D
    from scipy.special import erf
    dim = samples.shape[1]
    # 单维在 [-1, 1] 内的质量
    z_single = erf(1.0 / (sigma_global * np.sqrt(2)))
    log_Z_global = dim * np.log(z_single)

    # 3. 计算全局高斯校准对数概率
    inv_cov = 1.0 / (sigma_global ** 2)
    log_q_global_raw = -0.5 * (dim * np.log(2 * np.pi * (sigma_global ** 2)) + np.sum(samples ** 2, axis=1) * inv_cov)
    log_q_global_norm = log_q_global_raw - log_Z_global  # 减去 log_Z 进行校正

    # 4. 混合
    log_prob_ce = np.log(1 - alpha + 1e-15) + log_q_ce_norm
    log_prob_def = np.log(alpha + 1e-15) + log_q_global_norm

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


def failure_indicator(X_data, gp_model, gp_likelihood):
    """
    调用 GP 代理模型进行推理 (GPU加速)
    """
    # 转为 Tensor 并移动到 device
    x_tensor = torch.tensor(X_data, dtype=torch.float32).to("cuda")

    gp_model.eval(); gp_likelihood.eval()

    # 使用 GPyTorch 进行预测
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = gp_likelihood(gp_model(x_tensor))
        Y_pred = posterior.mean.cpu().numpy()
        sigma2 = posterior.variance.cpu().numpy()

    return Y_pred, np.sqrt(sigma2)


def sample_mixture_defensive(gmm, N_samples, alpha, sigma_global=0.8):
    """混合采样：(1-alpha) 来自 GMM，alpha 来自全局防御高斯"""
    n_ce = int(N_samples * (1 - alpha))
    n_def = N_samples - n_ce

    # GMM 采样
    samples_ce = []
    while len(samples_ce) < n_ce:
        new_s, _ = gmm.sample(n_ce * 2)
        mask = np.all((new_s >= -1) & (new_s <= 1), axis=1)
        samples_ce.extend(new_s[mask])
    samples_ce = np.array(samples_ce[:n_ce])

    # 全局高斯采样 (截断在 [-1, 1])
    samples_def = []
    while len(samples_def) < n_def:
        new_s = np.random.normal(0, sigma_global, size=(n_def * 2, gmm.means_.shape[1]))
        mask = np.all((new_s >= -1) & (new_s <= 1), axis=1)
        samples_def.extend(new_s[mask])
    samples_def = np.array(samples_def[:n_def])

    res = np.vstack([samples_ce, samples_def])
    np.random.shuffle(res)
    return res


def call_real_simulation(samples, runner):
    """调用CARLA真实仿真"""
    print(f"Running real simulation for {len(samples)} elite cases...")
    sim_results = []
    for idx in range(len(samples)):
        result, collision = runner.run(samples[idx].reshape(1, -1))
        sim_results.append(result)
    return np.array(sim_results).reshape(-1)


def update_surrogate_model(samples, true_values, gp_model, gp_likelihood):
    """更新代理模型数据"""
    print(f"Updating global surrogate model with {len(samples)} new cases...")

    # 转换为 GPU Tensor
    new_X = torch.tensor(samples, dtype=torch.float32).to("cuda")
    new_Y = torch.tensor(true_values, dtype=torch.float32).reshape(-1).to("cuda")

    old_X = gp_model.train_inputs[0]
    old_Y = gp_model.train_targets

    # 拼接数据
    train_x = torch.cat([old_X, new_X])
    train_y = torch.cat([old_Y, new_Y])

    # 设置新训练数据
    gp_model.set_train_data(inputs=train_x, targets=train_y, strict=False)

    # 保存
    save_path = '/home/hp/SENSE/rlsan/results/s5exp/last_updated_surrogate_model.pkl'
    save_surrogate_model(gp_model, gp_likelihood, save_path)
    print("Global model updated and saved.")

    # 确保数据留在 GPU
    gp_model.train_inputs = (train_x,)
    gp_model.train_targets = train_y
    gp_model.to("cuda"); gp_likelihood.to("cuda")
    if hasattr(gp_model, 'train_inputs') and gp_model.train_inputs is not None:
        gp_model.train_inputs = tuple(t.to("cuda") for t in gp_model.train_inputs)
    if hasattr(gp_model, 'train_targets') and gp_model.train_targets is not None:
        gp_model.train_targets = gp_model.train_targets.to("cuda")


def cross_entropy_refine_robust(q_init, gp_model, gp_likelihood, runner, N=10000, Ne_init=1000, eta = 0.5, T=10, lambda_mix=0.2):
    """
    集成 UCB 挖掘与防御性协方差膨胀的精炼过程
    """
    q_ce = q_init
    n_modes = q_init.n_components
    history = []

    for t in range(T):
        # --- Step 1: 混合采样 (GMM + Global Gaussian) ---
        X_samples = sample_mixture_defensive(q_ce, N, lambda_mix)

        # --- Step 2: UCB 评估 (解决 FN 问题) ---
        mu, sigma = failure_indicator(X_samples, gp_model, gp_likelihood)
        # 随迭代减少探索权重 kappa
        kappa = max(1.0, 4.0 * (1 - t/T))
        Y_ucb = mu + kappa * sigma

        # --- Step 3: Active Learning (保持不变) ---
        real_sim_indices = np.argsort(Y_ucb)[-100:]
        if len(real_sim_indices) > 0:
            Y_real = call_real_simulation(X_samples[real_sim_indices], runner)
            update_surrogate_model(X_samples[real_sim_indices], Y_real, gp_model, gp_likelihood)
            Y_ucb[real_sim_indices] = Y_real  # 真实值确定后，UCB 即为真实值

        # --- Step 4: 筛选精英并拟合 ---
        elite_idx = np.argsort(Y_ucb)[-Ne_init:]
        elites = X_samples[elite_idx]
        new_gmm = GaussianMixture(n_components=n_modes, covariance_type='full', random_state=42 + t).fit(elites)

        # --- Step 5: 【核心修改】防御性协方差膨胀 (Cov-Inflation) ---
        # 强制增加 min_eig 以对冲代理模型收缩过快
        min_eig = 0.15  # 相比之前的 0.08 进一步增加“厚度”
        for i in range(new_gmm.n_components):
            vals, vecs = np.linalg.eigh(new_gmm.covariances_[i])
            vals = np.clip(vals, min_eig, None)
            new_gmm.covariances_[i] = vecs @ np.diag(vals) @ vecs.T
        new_gmm.precisions_cholesky_ = _compute_precision_cholesky(new_gmm.covariances_, 'full')

        # --- Step 6: 动量更新 ---
        q_ce.means_ = (1 - eta) * q_ce.means_ + eta * new_gmm.means_
        q_ce.covariances_ = (1 - eta) * q_ce.covariances_ + eta * new_gmm.covariances_
        q_ce.weights_ = (1 - eta) * q_ce.weights_ + eta * new_gmm.weights_
        q_ce.precisions_cholesky_ = _compute_precision_cholesky(q_ce.covariances_, 'full')

        curr_score = np.mean(mu[elite_idx])
        history.append(curr_score)

        # --- Step 7: 增强型 ESS 监控与混合率动态调整 ---
        t_dim = q_ce.means_.shape[1]
        Z_t = estimate_truncation_constant(q_ce, t_dim, n_mc=100000)
        log_q_mix = log_q_mix_density_defensive(
            elites,
            q_ce,
            alpha=lambda_mix,
            Z_GMM=Z_t,
            sigma_global=0.8
        )
        log_p = log_target_distribution(elites)
        log_w = log_p - log_q_mix

        # 4. 数值稳定处理并计算 ESS
        w = np.exp(log_w - np.max(log_w))
        ESS_ratio = ((np.sum(w) ** 2) / np.sum(w ** 2)) / len(elites)

        # 【修改点 3】更激进的探索策略
        if ESS_ratio < 0.3:
            # 权重太集中，说明分布太窄，增加探索
            lambda_mix = min(0.75, lambda_mix + 0.1)
            print(f"  -> ESS Low ({ESS_ratio:.2%}), increasing lambda to {lambda_mix:.2f}")
        elif ESS_ratio > 0.7:
            # 权重分布均匀，说明分布很稳，减少探索 (收敛)
            lambda_mix = max(0.4, lambda_mix - 0.05)
            print(f"  -> ESS High ({ESS_ratio:.2%}), decreasing lambda to {lambda_mix:.2f}")
        else:
            print(f"  -> ESS Moderate ({ESS_ratio:.2%}), keeping lambda at {lambda_mix:.2f}")

        print(f"Iter {t + 1}: Elite Mean Score = {curr_score:.4f} | Kappa = {kappa:.2f}, Mix_Rate={lambda_mix:.2f}")


    return q_ce, lambda_mix, history


if __name__ == "__main__":
    # 参数解析 (CarlaRunner 需要)
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='rlsan')
    parser.add_argument('--output_dir', type=str, default='log')
    parser.add_argument('--ROOT_DIR', type=str,default=r"/home/hp/SENSE")
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
    parser.add_argument('--port', type=int, default=2004, help='port to communicate with carla')
    parser.add_argument('--tm_port', type=int, default=8004, help='traffic manager port')
    parser.add_argument('--fixed_delta_seconds', type=float, default=0.1)

    # 添加其他必要的 Carla 参数...
    args = parser.parse_args()
    args_dict = vars(args)
    set_torch_variable(args.device)
    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    # 1. 加载 Surrogate GP
    gp_path = '../../results/s5exp/surrogate_model.pkl'
    gp_model, gp_likelihood = load_surrogate_model(gp_path)
    gp_model.to(args.device); gp_likelihood.to(args.device)
    print("GP Model loaded.")

    # 加载 TDA-Informed GMM
    tda_model_path = '../../results/s5exp/tda_gmm_model.pkl'
    try:
        q_rl, k_optimal = load_tda_model(tda_model_path)
        print(f"TDA-GMM loaded. Components: {k_optimal}, Means shape: {q_rl.means_.shape}")
    except FileNotFoundError:
        print("Error: TDA model not found. Please run the TDA analysis script first.")

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
    lambda_mix_init = 0.3
    final_q_ce, final_lambda_mix, history = cross_entropy_refine_robust(
        q_rl, gp_model, gp_likelihood, runner,
        N=10000, Ne_init=1000, eta=0.5, T=15, lambda_mix=lambda_mix_init
    )

    # 截断校准 ---
    dim = final_q_ce.means_.shape[1]
    final_Z = estimate_truncation_constant(final_q_ce, dim)

    # 4. 生成最终评估样本 (IS Evaluation)
    print("Generating samples for final IS estimation...")
    N_total = 10000
    X_final = sample_mixture_defensive(final_q_ce, N_total, alpha=final_lambda_mix, sigma_global=0.8)
    np.random.shuffle(X_final)

    # 计算【校准后】的 Log 概率
    log_p_values = log_target_distribution(X_final)
    log_q_mix_values = log_q_mix_density_defensive(X_final, final_q_ce, alpha=final_lambda_mix, Z_GMM=final_Z, sigma_global=0.8)

    # 6. 保存最终精炼结果（含完整模型状态，解决版本冲突）
    save_package = {
        'gmm': final_q_ce,
        'lambda_mix': final_lambda_mix,
        'Z_constant': final_Z,
        'history': history,
        'q_mix_values': np.exp(log_q_mix_values),  # 保存 exp 值用于直观查看
        'log_q_mix_values': log_q_mix_values  # 保存 log 值用于精确计算
    }

    # optional: 为了验证代理模型的精确性,另存重要性分布的样本并进行实际仿真
    with open('../../results/s5exp/final_refined_IS_package_20251222.pkl', "wb") as f:
        pickle.dump(save_package, f)
    with open('../../results/IS/scenario05/scenario05_20251222/sampled_parameters.pkl', "wb") as f:
        pickle.dump(X_final, f)
    print(f"All data calibrated and saved with Z={final_Z:.4f}")

    # 8. 绘制历史
    plt.figure()
    plt.plot(history, marker='o')
    plt.title("CE Refinement History")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Elite Prob")
    plt.savefig('../../results/s5exp/ce_history.png')
    plt.close()