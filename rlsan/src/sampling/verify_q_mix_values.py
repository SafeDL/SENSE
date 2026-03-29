"""
Importance Sampling Pre-Simulation Diagnostic Tool
用于在真实仿真前验证替代分布 q_mix 的合理性与统计一致性
"""
import numpy as np
import pickle
import os.path as osp
import matplotlib.pyplot as plt


def load_data(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


def get_log_p(x):
    """计算目标分布（均匀分布）的对数概率密度"""
    dim = x.shape[1]
    vol = 2.0 ** dim  # [-1, 1]^3 -> vol = 8
    return np.full(x.shape[0], -np.log(vol))


def run_pre_simulation_diagnostics(package_path, params_path):
    """
    执行三维诊断：权重一致性、有效样本量 (ESS)、代理模型估计值对比
    """
    print("=" * 50)
    print("STEP 1: Loading Refined IS Package...")
    package = load_data(package_path)
    X_final = load_data(params_path)

    # 提取关键参数
    log_q_mix = package['log_q_mix_values']
    Z_val = package['Z_constant']
    lambda_mix = package['lambda_mix']

    # 重新计算 log_p 以确保量纲一致
    log_p = get_log_p(X_final)

    # 计算似然比权重 w = p(x) / q(x)
    # 使用 exp(log_p - log_q) 保证数值稳定性
    weights = np.exp(log_p - log_q_mix)
    N = len(weights)

    print("\nSTEP 2: Statistical Consistency Analysis")
    # 1. 权重均值 (理论上必须等于 1.0)
    weight_mean = np.mean(weights)
    # 2. 有效样本量 ESS (衡量权重分布的均匀性)
    ess = (np.sum(weights) ** 2) / np.sum(weights ** 2)
    # 3. 权重极值比 (衡量是否存在“权重爆炸”点)
    weight_max_min_ratio = np.max(weights) / (np.min(weights) + 1e-20)
    # 4. 增加 C.V. 诊断
    weight_cv = np.std(weights) / np.mean(weights)

    print(f"-> Weight Mean: {weight_mean:.4f} (理想值: 1.0000)")
    print(f"-> ESS: {ess:.1f} / {N} ({ess / N * 100:.1f}%)")
    print(f"-> Max/Min Weight Ratio: {weight_max_min_ratio:.2e}")
    print(f"-> Weight C.V.: {weight_cv:.4f} (理想值 < 2.0, 风险阈值 > 5.0)")


    print("\nSTEP 3: Surrogate-based Failure Rate Estimate")
    # 这里模拟指示函数 (假设你已经有代理模型的初步预测，或者直接从 package 获取)
    # 我们利用代理模型的预测结果进行加权估计
    # 注意：Script 1 运行完毕时，importance_sampling_estimate 已经给出了基于 GP 的估计值
    # 如果该值与 0.033225 偏差巨大，说明 q_mix 存在系统性偏置。

    # --- 可视化诊断图 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：对数权重分布 (Log-Weights Distribution)
    # 理想状态应呈现集中且对称的钟形，如果存在超长尾部，则说明存在极少数点绑架结果的风险
    axes[0].hist(np.log10(weights), bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel("Log10(Weight)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Importance Weights (Log Scale)")
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # 右图：有效样本量随样本增加的趋势
    ess_history = [(np.sum(weights[:i]) ** 2) / np.sum(weights[:i] ** 2) for i in range(100, N + 1, 100)]
    axes[1].plot(range(100, N + 1, 100), ess_history, label='ESS Path', color='orange', lw=2)
    axes[1].axhline(y=N * 0.1, color='red', ls='--', label='Critical Line (10% N)')
    axes[1].set_xlabel("Sample Size")
    axes[1].set_ylabel("ESS")
    axes[1].set_title("Effective Sample Size Growth")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 50)
    print("FINAL RECOMMENDATION:")

    # 综合判定逻辑
    decision_flags = []
    if abs(weight_mean - 1.0) > 0.1:
        decision_flags.append("RED: Normalization failure (Check Z or Lambda mismatch)!")
    if ess < N * 0.05:
        decision_flags.append("ORANGE: Severe weight collapse! Result will be dominated by few samples.")
    if weight_max_min_ratio > 1e6:
        decision_flags.append("YELLOW: High variance detected! Consider increasing lambda_mix.")

    if not decision_flags:
        print(">>> [GREEN] q_mix seems robust. Proceed to CARLA simulation.")
    else:
        for flag in decision_flags:
            print(f">>> {flag}")
    print("=" * 50)


if __name__ == "__main__":
    np.random.seed(42)
    # 路径配置
    params_file = osp.join('../../results/IS/scenario01', 'sampled_parameters.pkl')
    package_file = '../../results/s1exp/final_refined_IS_package.pkl'

    package = load_data(package_file)
    X_data = load_data(params_file)

    # 核心诊断：检查 package 里的 Z 和 lambda
    Z_saved = package.get('Z_constant', 1.0)
    lambda_saved = package.get('lambda_mix', 0.25)
    print(f"Loaded Package: Z={Z_saved:.4f}, Lambda={lambda_saved:.4f}")

    if osp.exists(package_file) and osp.exists(params_file):
        run_pre_simulation_diagnostics(package_file, params_file)
    else:
        print("Error: Required pkl files not found. Please run Script 1 first.")