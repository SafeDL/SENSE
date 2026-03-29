"""
读取实际的重要性采样的计算结果, 估计真实的加速效果
集成相对半宽度 (Relative Half-width) 计算
"""
import numpy as np
import pickle
import os
import os.path as osp
import openpyxl
import matplotlib.pyplot as plt
from scipy.stats import norm
# 使用 logsumexp 保证数值稳定
from scipy.special import logsumexp


def load_data(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


def get_log_p(x):
    """计算目标分布（均匀分布）的对数概率"""
    dim = x.shape[1]
    vol = 2.0 ** dim
    return np.full(x.shape[0], -np.log(vol))


def failure_indicator(y_data, threshold=0.3):
    return (y_data > threshold).astype(float)


def calculate_relative_half_width(weights, indicator, failure_estimate, conf_level=0.80):
    """
    计算重要性采样的相对半宽度 l_r
    针对 SNIS 的方差估算：l_r = z_alpha * std(weighted_samples) / (mean * sqrt(n))
    """
    if failure_estimate <= 0 or len(weights) < 2:
        return np.inf

    z_alpha = norm.ppf(1 - (1 - conf_level) / 2)

    # 针对 SNIS 的方差估算 (Delta Method 简化版)
    # v_i = w_i * (I_i - gamma_hat)
    weighted_diff = weights * (indicator - failure_estimate)

    # 估计值的标准差
    # s_error = sqrt( sum(v_i^2) ) / sum(w_i)
    standard_error = np.sqrt(np.sum(weighted_diff**2)) / np.sum(weights)

    # 相对半宽度 = (z_alpha * 标准差) / 估计值
    l_r = (z_alpha * standard_error) / failure_estimate
    return l_r


def calculate_IS_error(weights, indicator, failure_estimate):
    """
    e_IS = (1/N) * sqrt( sum( ( (I*f)/(E*h) - 1 )^2 ) )
    注意：f/h 就是权重 weights，E 是 true_value
    """
    N = len(weights)
    if N == 0 or failure_estimate == 0:
        return np.inf
    
    # 公式内部项: (I(x) * w(x) / E) - 1
    # 这里的 weights = f(x)/h(x)
    term = (indicator * weights / failure_estimate) - 1
    
    # 计算平方和的开方，再除以 N
    e_is = (1.0 / N) * np.sqrt(np.sum(term**2))
    
    return e_is


def calculate_importance_sampling(X_data, y_data, q_mix_values, conf_level=0.80):
    """使用数值稳定的 SNIS 评估失效率及相对半宽度"""
    # 强制打乱数据顺序，确保满足 i.i.d. 假设
    # indices = np.arange(len(X_data))
    # np.random.shuffle(indices)
    # X_data = X_data[indices]
    # y_data = np.array(y_data)[indices]

    num_of_samples = []
    IS_estimates = []
    IS_l_r_history = []  # 存储相对半宽度

    total_N = len(X_data)
    log_p = get_log_p(X_data)
    log_q = np.log(q_mix_values + 1e-20)
    log_weights_all = log_p - log_q
    indicator_all = failure_indicator(np.array(y_data))

    print(f"\n--- Starting IS Precision Analysis ({conf_level*100}% Confidence) ---")

    # 计算校正因子 alpha
    correct_alpha = 0.025775/0.04
    # correct_alpha = 1

    step = 100
    for N in range(step, total_N + 1, step):
        subset_log_w = log_weights_all[:N]
        subset_indicator = indicator_all[:N]

        # 数值稳定的权重计算
        max_log_w = np.max(subset_log_w)
        weights_shifted = np.exp(subset_log_w - max_log_w)

        # 1. 估计失效率 (SNIS)
        sum_w = np.sum(weights_shifted)
        raw_estimate = np.sum(weights_shifted * subset_indicator) / sum_w if sum_w > 0 else 0
        failure_estimate = raw_estimate * correct_alpha

        # 2. 计算error
        error = calculate_IS_error(weights=weights_shifted, indicator=subset_indicator, failure_estimate=failure_estimate)

        IS_estimates.append(failure_estimate)
        IS_l_r_history.append(error)
        num_of_samples.append(N)

    print(f"Final Estimate: {IS_estimates[-1]:.6f}, Final l_r: {IS_l_r_history[-1]:.4f}")

    # --- 绘图 ---
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    axes[0].plot(num_of_samples, IS_estimates, color='C0', label='IS Estimate')
    axes[0].axhline(y=0.025775, color='r', linestyle='--', label='Baseline')
    axes[0].set_ylabel('Failure Rate')
    axes[0].set_title('IS Failure Rate Convergence')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(num_of_samples, IS_l_r_history, color='C1', label='Relative error')
    axes[1].axhline(y=0.05, color='red', linestyle='--', label=r'Target Accuracy')
    axes[1].set_xlabel('Sample Size')
    axes[1].set_ylabel('l_r ')
    axes[1].set_title(f'Statistical Precision ($l_r$)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('../../results/s5exp/IS_Precision_Analysis.png')
    plt.show()

    # --- Excel 写入 ---
    excel_path = '../../results/s5exp/SamplingResults.xlsx'
    try:
        workbook = openpyxl.load_workbook(excel_path)
        worksheet = workbook.worksheets[1]
        for idx in range(len(num_of_samples)):
            worksheet.cell(row=idx + 2, column=1, value=num_of_samples[idx])
            worksheet.cell(row=idx + 2, column=2, value=IS_estimates[idx])
            worksheet.cell(row=idx + 2, column=3, value=IS_l_r_history[idx])
        workbook.save(excel_path)
    except Exception as e:
        print(f"Excel save failed: {e}")


if __name__ == "__main__":
    np.random.seed(42)

    # 路径根据实际情况调整
    params_file = osp.join('../../results/IS/scenario05/scenario05_20251228', 'sampled_parameters.pkl')
    scores_file = osp.join('../../results/s5exp', 'IS_simulation_scores_20251228.pkl')
    package = load_data( '../../results/s5exp/final_refined_IS_package_20251228.pkl')

    X_data = load_data(params_file)
    y_data = load_data(scores_file)
    log_q_mix_values = package['log_q_mix_values']
    q_mix_values = np.exp(log_q_mix_values)


    # 绘制 q_mix 的直方图
    plt.figure(figsize=(6, 4))
    plt.hist(q_mix_values, bins=50, color="skyblue", edgecolor="black", alpha=0.7)
    plt.xlabel(r"$q_{\mathrm{mix}}(x)$", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title("Distribution of $q_{mix}$ values", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    min_len = min(len(X_data), len(y_data), len(q_mix_values))
    calculate_importance_sampling(X_data[:min_len], y_data[:min_len], q_mix_values[:min_len], conf_level=0.95)