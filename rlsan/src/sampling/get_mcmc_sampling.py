"""
在指定的测试空间内进行大规模蒙特卡洛采样, 并计算失效率估计值及其误差
"""
import numpy as np
import os
import openpyxl
import pickle
import gpytorch
import torch
import matplotlib.pyplot as plt
from rlsan.src.surrogate.utils import load_surrogate_model
from scipy.stats import norm  # 用于计算分位数


def failure_indicator(X_data, threshold=0.3):
    """
    调用 gp_model 进行推理
    """
    # 转为 Tensor 并移动到 device
    x_tensor = torch.tensor(X_data, dtype=torch.float32).to("cuda")

    # 使用 GPyTorch 进行预测
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = gp_likelihood(gp_model(x_tensor))
        Y_pred = posterior.mean.cpu().numpy()

    # 示性函数,1代表碰撞,0代表安全测试用例
    indicator = (Y_pred > threshold).astype(float)
    return indicator


def calculate_mcmc_sampling(N=500000, conf_level=0.80):
    """
    逐批次执行蒙特卡洛采样，计算失效率估计值及其相对半宽度 l_r
    """
    num_of_samples = []  # 记录当前考察的样本数量
    mcmc_estimates = []  # 记录不同样本下mcmc采样估计失效率
    mcmc_l_r_history = []  # 替换原来的 error 列表

    # 设定置信度分位数 z_alpha (80% 置信度下约为 1.28)
    z_alpha = norm.ppf(1 - (1 - conf_level) / 2)
    print(f"Confidence Level: {conf_level * 100}%, z_alpha: {z_alpha:.3f}")

    mcmc_samples = np.random.uniform(-1, 1, size=(N, 3))

    # (optional) 另存蒙特卡洛的结果进行实际仿真
    with open('../../results/MCMC/scenario08/sampled_parameters.pkl', "wb") as f:
        pickle.dump(np.vstack(mcmc_samples), f)
    print(f"MCMC samples saved")

    # 依次从samples中逐次选取批次样本进行评估
    for i in range(1000, len(mcmc_samples)+1, 1000):
        # 读取 0 到 i 的数据
        subset_samples = mcmc_samples[:i]
        test_restuls = failure_indicator(subset_samples)
        failure_estimate = np.mean(test_restuls)

        # 计算相对半宽度 l_r [依据公式: z_alpha * sqrt((1-gamma)/(n*gamma))]
        # 只有在发现失效样本且估计值不为 0 时，l_r 才有定义
        if failure_estimate > 0:
            l_r = z_alpha * np.sqrt((1 - failure_estimate) / (i * failure_estimate))
        else:
            l_r = np.inf  # 未发现失效前，误差无穷大

        mcmc_l_r_history.append(l_r)
        num_of_samples.append(len(subset_samples))
        mcmc_estimates.append(failure_estimate)

        if i % 10000 == 0:
            print(f"Progress: {i}/{N}, Estimate: {failure_estimate:.6f}, l_r: {l_r:.4f}")


    fig, axes = plt.subplots(2, 1, figsize=(8, 12))
    axes[0].plot(num_of_samples, mcmc_estimates, '-o', color='C0', label='mcmc estimate')
    axes[0].set_xlabel('Number of samples')
    axes[0].set_ylabel('Estimated failure rate')
    axes[0].set_title('mcmc estimates vs sample size')
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(num_of_samples, mcmc_l_r_history, '-s', color='C1', label='Relative Half-width')
    axes[1].axhline(y=0.2, color='r', linestyle='--', label='Convergence Target')
    axes[1].set_ylim(0, 1.0)
    axes[1].set_xlabel('Number of samples')
    axes[1].set_ylabel('Estimation error')
    axes[1].set_title('mcmc estimation error vs sample size')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)

    # 加载代理模型
    model_path = '../../results/s1exp/surrogate_model.pkl'
    gp_model, gp_likelihood = load_surrogate_model(model_path)
    gp_model.to("cuda"); gp_likelihood.to("cuda")
    print("Global GP Model loaded successfully.")

    # 在给定测试空间内大规模采样,然后计算蒙特卡罗失效率
    calculate_mcmc_sampling(N=100000)

