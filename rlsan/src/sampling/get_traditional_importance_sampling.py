import os.path as osp
import pickle
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from safebench.util.torch_util import set_seed, set_torch_variable
from safebench.util.run_util import load_config
from safebench.carla_runner_simple import CarlaRunner

def log_target_distribution(x):
    """目标空间 [-1, 1]^D 的均匀分布对数密度"""
    vol = 2.0 ** x.shape[1]
    return np.full(x.shape[0], -np.log(vol))

def estimate_truncation_constant(gmm, n_mc=500000):
    """估算 Z 保证 IS 无偏性，增加样本量以减少基准误差"""
    samples_raw, _ = gmm.sample(n_mc)
    valid_mask = np.all((samples_raw >= -1) & (samples_raw <= 1), axis=1)
    Z = np.mean(valid_mask)
    return max(Z, 1e-6)

def sample_truncated_gmm(gmm, N_samples):
    """边界约束采样"""
    samples = []
    while len(samples) < N_samples:
        new_samples, _ = gmm.sample(int(N_samples * 2))
        mask = np.all((new_samples >= -1) & (new_samples <= 1), axis=1)
        samples.extend(new_samples[mask])
    return np.array(samples[:N_samples])

def fit_best_gmm_bic(X_elites, k_range=range(1, 10)):
    """利用 BIC 准则自动寻找最优模态数 K"""
    best_bic = np.inf
    best_gmm = None

    for k in k_range:
        if len(X_elites) <= k * 5: continue
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42).fit(X_elites)
        bic = gmm.bic(X_elites)
        if bic < best_bic:
            best_bic, best_gmm = bic, gmm

    # 协方差膨胀保护 (Defensive Inflation)
    min_eig = 0.08
    for i in range(best_gmm.n_components):
        vals, vecs = np.linalg.eigh(best_gmm.covariances_[i])
        best_gmm.covariances_[i] = vecs @ np.diag(np.clip(vals, min_eig, None)) @ vecs.T
    best_gmm.precisions_cholesky_ = _compute_precision_cholesky(best_gmm.covariances_, 'full')
    return best_gmm


def cross_entropy_method_adaptive_k(eval_func, dim=3, N_iter=10, N_samples=200, elite_frac=0.2, eta=0.7):
    """
    自适应 K 值的经典 CEM 搜索
    """
    # 初始状态：单峰广域覆盖
    curr_gmm = GaussianMixture(n_components=1, covariance_type='full')
    curr_gmm.means_ = np.zeros((1, dim))
    curr_gmm.covariances_ = np.array([np.eye(dim) * 0.8])
    curr_gmm.weights_ = np.array([1.0])
    curr_gmm.precisions_cholesky_ = _compute_precision_cholesky(curr_gmm.covariances_, 'full')

    history = []
    print(f"\n>>> Starting Adaptive-K CEM | Budget: {N_iter * N_samples} Simulations")

    for t in range(N_iter):
        # A. 物理采样
        X_samples = sample_truncated_gmm(curr_gmm, N_samples)

        # B. 真实物理仿真评估
        Y_scores = eval_func(X_samples)

        # C. 精英筛选
        Ne = max(int(N_samples * elite_frac), 40)
        elite_idx = np.argsort(Y_scores)[-Ne:]
        X_elites = X_samples[elite_idx]

        # D. 自动模态发现与拟合
        new_gmm = fit_best_gmm_bic(X_elites)

        # E. 参数平滑更新 (若 K 发生变化则直接替换)
        if new_gmm.n_components == curr_gmm.n_components:
            curr_gmm.means_ = (1 - eta) * curr_gmm.means_ + eta * new_gmm.means_
            curr_gmm.weights_ = (1 - eta) * curr_gmm.weights_ + eta * new_gmm.weights_
            curr_gmm.covariances_ = (1 - eta) * curr_gmm.covariances_ + eta * new_gmm.covariances_
            curr_gmm.precisions_cholesky_ = _compute_precision_cholesky(curr_gmm.covariances_, 'full')
        else:
            print(f"  Iter {t + 1}: Structural Shift K -> {new_gmm.n_components}")
            curr_gmm = new_gmm

        m_score = np.mean(Y_scores[elite_idx])
        history.append(m_score)
        print(f"  Iter {t + 1}/{N_iter}: Elite Mean Score = {m_score:.4f} | K = {curr_gmm.n_components}")

    return curr_gmm, history


if __name__ == "__main__":
    # 参数解析 (CarlaRunner 需要)
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='cross_entropy_baselines')
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

    # 2. 初始化 CARLA Runner (仅用于 Active Learning)
    agent_config_path = osp.join(args.ROOT_DIR, 'safebench/agent/config', args.agent_cfg)
    agent_config = load_config(agent_config_path)
    agent_config.update(args_dict)

    # load scenario config
    scenario_config_path = osp.join(args.ROOT_DIR, 'rlsan/config', args.scenario_cfg)
    scenario_config = load_config(scenario_config_path)
    scenario_config.update(args_dict)  # 将在本脚本中指定的参数更新到配置中
    runner = CarlaRunner(agent_config, scenario_config, step_by_step=True)

    print("\n>>> Starting [Real-Simulation CEM] Baseline (Expensive!)...")
    def eval_real(x):
        # 逐个运行仿真
        return np.array([runner.run(s.reshape(1, -1))[0] for s in x]).flatten()
    
    final_q, history = cross_entropy_method_adaptive_k(eval_real, N_iter=10, N_samples=200)

    # 校准 Z
    final_Z = estimate_truncation_constant(final_q)

    # 生成 10,000 个最终评估样本
    N_total = 10000
    X_final = sample_truncated_gmm(final_q, N_total)
    np.random.shuffle(X_final)

    # 计算校准后的 Log 概率 (这里 CEM 不带混合项，体现其原始特性)
    log_p = log_target_distribution(X_final)
    log_q = final_q.score_samples(X_final) - np.log(final_Z)

    save_package = {
        'gmm': final_q,
        'Z_constant': final_Z,
        'history': history,
        'log_q_mix_values': log_q,
        'q_mix_values': np.exp(log_q)
    }

    with open('../../results/s5exp/baseline_refined_IS_package.pkl', "wb") as f:
        pickle.dump(save_package, f)
    with open(osp.join('../../results/IS/scenario05/cross_entropy/sampled_parameters.pkl'), "wb") as f:
        pickle.dump(X_final, f)

    # 绘图对比收敛
    plt.figure(figsize=(10, 5))
    plt.plot(history, label='Real-Sim-Guided CEM', marker='o')
    plt.title("Convergence Comparison: Surrogate vs Real-Sim")
    plt.xlabel("Iteration")
    plt.ylabel("Elite Mean Score")
    plt.legend()
    plt.savefig('../../results/s5exp/baseline_comparison.png')
    
    print("\nBaselines completed. Results saved.")