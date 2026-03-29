"""
Advanced Physics-Anchored Sequential Importance Sampling (APA-SIS)
特性：
1. 物理真值驱动：丢弃 GP 代理模型，直接利用 CARLA 仿真得分驱动分布进化。
3. 拓扑先验注入：复用 TDA-GMM 作为初始分布，解决经典 CE 搜索盲目性。
4. 鲁棒性防御：强制 min_eig = 0.15 并结合 ESS 动态调节 lambda_mix。
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

from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from scipy.special import logsumexp

# 基础配置与导入 (请根据实际路径调整)
from safebench.util.torch_util import set_seed, set_torch_variable
from safebench.util.run_util import load_config
from safebench.carla_runner_simple import CarlaRunner

warnings.filterwarnings("ignore")


# ================= 核心数学工具 =================

def calculate_ess(log_weights):
    """计算有效样本量比例 (ESS Ratio)"""
    weights = np.exp(log_weights - np.max(log_weights))
    ess = (np.sum(weights) ** 2) / np.sum(weights ** 2)
    return ess / len(log_weights)


def estimate_truncation_constant(gmm, dim, n_mc=1000000):
    """估算 GMM 在 [-1, 1]^D 盒子内的截断常数 Z"""
    print(f"Estimating truncation constant Z for GMM (dim={dim})...")
    samples_raw, _ = gmm.sample(n_mc)
    valid_mask = np.all((samples_raw >= -1) & (samples_raw <= 1), axis=1)
    Z = np.mean(valid_mask)
    return max(Z, 1e-6)


def log_target_distribution(x):
    """目标均匀分布对数密度 log p(x)"""
    dim = x.shape[1]
    return np.full(x.shape[0], -dim * np.log(2.0))


def log_q_mix_density_calibrated(samples, gmm, lambda_mix, Z_constant):
    """计算校准后的混合密度: log( (1-λ)*(q/Z) + λp )"""
    log_q_norm = gmm.score_samples(samples) - np.log(Z_constant)
    log_p = log_target_distribution(samples)
    return logsumexp(np.vstack([
        np.log(1 - lambda_mix + 1e-15) + log_q_norm,
        np.log(lambda_mix + 1e-15) + log_p
    ]).T, axis=1)


# ================= GMM 稳健拟合组件 =================

def fit_weighted_robust_gmm(X_elites, k_tda):
    """基于物理分数的加权拟合 + 协方差强制膨胀"""
    # 动态搜索最优 K
    k_range = range(max(1, k_tda - 3), k_tda + 3)
    best_bic, best_gmm = np.inf, None

    for k in k_range:
        if len(X_elites) < k * 3: continue
        gmm = GaussianMixture(n_components=k, covariance_type='full', reg_covar=1e-4, random_state=42)
        gmm.fit(X_elites)
        bic = gmm.bic(X_elites)
        if bic < best_bic:
            best_bic, best_gmm = bic, gmm

    # --- 核心防御：强制 min_eig = 0.10 对冲边界噪声 ---
    min_eig = 0.10
    for i in range(best_gmm.n_components):
        vals, vecs = np.linalg.eigh(best_gmm.covariances_[i])
        best_gmm.covariances_[i] = vecs @ np.diag(np.clip(vals, min_eig, None)) @ vecs.T
    best_gmm.precisions_cholesky_ = _compute_precision_cholesky(best_gmm.covariances_, 'full')

    return best_gmm


def sample_defensive_mixture(gmm, N_samples, lambda_mix):
    """防御性混合采样"""
    n_exploit = int(N_samples * (1 - lambda_mix))
    n_explore = N_samples - n_exploit

    samples_gmm = []
    while len(samples_gmm) < n_exploit:
        new_s, _ = gmm.sample(n_exploit * 3)
        mask = np.all((new_s >= -1) & (new_s <= 1), axis=1)
        samples_gmm.extend(new_s[mask])

    np.random.shuffle(samples_gmm)
    samples_uni = np.random.uniform(-1, 1, size=(n_explore, gmm.means_.shape[1]))
    res = np.vstack([np.array(samples_gmm[:n_exploit]), samples_uni])
    np.random.shuffle(res)
    return res


# ================= 主循环 =================
def sequential_physical_refine(q_init, k_tda, runner, N_batch=200, T=10, lambda_init=0.3, eta=0.5):
    """
    APA-SIS 物理标定循环：丢弃代理模型，直接利用真实分数
    """
    q_current = q_init
    lambda_mix = lambda_init

    # 全局池：用于跨迭代保留真实物理失效点
    pool_x = np.empty((0, q_init.means_.shape[1]))
    pool_y = np.empty((0,))
    history = []

    print(f"\n>>> APA-SIS Physics Refinement | Initial K={k_tda} | Seed Mixed Rate={lambda_init}")

    for t in range(T):
        # 1. 混合采样
        X_batch = sample_defensive_mixture(q_current, N_batch, lambda_mix)

        # 2. 真实物理仿真 (CARLA 直接运行)
        Y_real = np.array([runner.run(x.reshape(1, -1))[0] for x in X_batch]).flatten()

        # 3. 精英池维护
        pool_x = np.vstack([pool_x, X_batch])
        pool_y = np.concatenate([pool_y, Y_real])
        n_elite = int(len(pool_y) * 0.15)

        # 稳健性检查：确保样本量足以拟合 K 个高斯分量 (至少 10*K)
        n_elite = max(n_elite, k_tda * 10)

        # 排序并选取前 n_elite 个样本
        elite_indices = np.argsort(pool_y)[-n_elite:]
        X_elites = pool_x[elite_indices]

        # 4. 稳健拟合
        new_gmm = fit_weighted_robust_gmm(X_elites, k_tda)
        curr_Z = estimate_truncation_constant(new_gmm, dim=3)

        # 5. ESS 监控与 lambda_mix 调节
        X_diag = sample_defensive_mixture(new_gmm, N_samples=20000, lambda_mix=lambda_mix)
        log_q_diag = log_q_mix_density_calibrated(X_diag, new_gmm, lambda_mix, curr_Z)
        log_p_diag = log_target_distribution(X_diag)
        ess_ratio = calculate_ess(log_p_diag - log_q_diag)

        # 动态防御：若 ESS 太低，说明分布在噪声边界处收缩过快
        if ess_ratio < 0.30:
            lambda_mix = min(0.75, lambda_mix + 0.10)
        elif ess_ratio > 0.70:
            lambda_mix = max(0.30, lambda_mix - 0.05)

        # 6. 参数平滑更新 (Momentum)
        if new_gmm.n_components == q_current.n_components:
            q_current.means_ = (1 - eta) * q_current.means_ + eta * new_gmm.means_
            q_current.covariances_ = (1 - eta) * q_current.covariances_ + eta * new_gmm.covariances_
            q_current.weights_ = (1 - eta) * q_current.weights_ + eta * new_gmm.weights_
            q_current.precisions_cholesky_ = _compute_precision_cholesky(q_current.covariances_, 'full')
        else:
            q_current = new_gmm

        # 池子管理：保留最近 3-4 轮的样本，过大则移除陈旧数据
        max_pool = N_batch * 4
        if len(pool_x) > max_pool:
            pool_x, pool_y = pool_x[-max_pool:], pool_y[-max_pool:]

        batch_m = np.mean(Y_real)
        history.append(batch_m)
        print(
            f"Iter {t + 1}: Fail_Points={len(X_elites)} | Batch_Mean={batch_m:.4f} | ESS={ess_ratio:.2%} | Lambda={lambda_mix:.2f}")

    return q_current, lambda_mix, history


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
    parser.add_argument('--port', type=int, default=2000, help='port to communicate with carla')
    parser.add_argument('--tm_port', type=int, default=8000, help='traffic manager port')
    parser.add_argument('--fixed_delta_seconds', type=float, default=0.1)

    # 添加其他必要的 Carla 参数...
    args = parser.parse_args()
    args_dict = vars(args)
    set_torch_variable(args.device)
    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    # 1. 加载 TDA 初始化
    tda_path = '../../results/s5exp/tda_gmm_model.pkl'
    with open(tda_path, 'rb') as f:
        tda_data = pickle.load(f)
    q_rl, k_optimal = tda_data['gmm'], tda_data['tda_k']

    # 2. 初始化物理仿真环境
    agent_config_path = osp.join(args.ROOT_DIR, 'safebench/agent/config', args.agent_cfg)
    agent_config = load_config(agent_config_path)
    agent_config.update(args_dict)

    # load scenario config
    scenario_config_path = osp.join(args.ROOT_DIR, 'rlsan/config', args.scenario_cfg)
    scenario_config = load_config(scenario_config_path)
    scenario_config.update(args_dict)
    runner = CarlaRunner(agent_config, scenario_config, step_by_step=True)


    # 3. 执行纯物理精炼 (丢弃代理模型)
    # 注意：TDA 种子样本已经拟合在 q_rl 中
    final_q, final_lambda, score_history = sequential_physical_refine(
        q_init=q_rl, k_tda=k_optimal, runner=runner,  # 此处替换为你的 runner
        N_batch=1000, T=10, lambda_init=0.5, eta=0.4
    )

    # 4. 最终校准与打包
    final_Z = estimate_truncation_constant(final_q, dim=3)

    # 生成 20,000 个最终评估样本
    X_final = sample_defensive_mixture(final_q, 20000, final_lambda)
    log_q_final = log_q_mix_density_calibrated(X_final, final_q, final_lambda, final_Z)

    save_package = {
        'gmm': final_q,
        'lambda_mix': final_lambda,
        'Z_constant': final_Z,
        'history': score_history,
        'log_q_mix_values': log_q_final,
        'q_mix_values': np.exp(log_q_final),
    }

    with open('../../results/s5exp/apa_sis_refined_package.pkl', "wb") as f:
        pickle.dump(save_package, f)
    with open('../../results/IS/scenario05/scenario05_20251227/sampled_parameters.pkl', "wb") as f:
        pickle.dump(X_final, f)

    print(f"\n[SUCCESS] APA-SIS Package Saved!")