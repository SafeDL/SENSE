import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def calculate_metrics(mat_file):
    """计算关键指标"""
    data = sio.loadmat(mat_file)

    # Asymptotic Return: 最后100个episode的平均奖励
    mean_rewards = data['mean_rewards'].flatten()
    asymptotic_return = np.mean(mean_rewards[-100:])

    # Action Variance: 所有动作的方差
    action_vars = []
    for key in ['action_c1', 'action_c2', 'action_vs', 'action_w']:
        if key in data:
            action_vars.append(np.var(data[key]))
    action_variance = np.mean(action_vars) if action_vars else 0

    # Episodes to Convergence: 奖励稳定时的episode数
    threshold = np.max(mean_rewards) * 0.95
    convergence_idx = np.where(mean_rewards >= threshold)[0]
    episodes_to_convergence = convergence_idx[0] if len(convergence_idx) > 0 else len(mean_rewards)

    return {
        'asymptotic_return': asymptotic_return,
        'action_variance': action_variance,
        'episodes_to_convergence': episodes_to_convergence
    }

# 分析 DDPG 数据
print("DDPG Multi-Run 数据:")
metrics_multi = calculate_metrics('ddpg_multi_run_data.mat')
print(f"  Asymptotic Return: {metrics_multi['asymptotic_return']:.4f}")
print(f"  Action Variance: {metrics_multi['action_variance']:.6f}")
print(f"  Episodes to Convergence: {metrics_multi['episodes_to_convergence']}")

print("\nDDPG Run 1 数据:")
metrics_run1 = calculate_metrics('ddpg_run_1.mat')
print(f"  Asymptotic Return: {metrics_run1['asymptotic_return']:.4f}")
print(f"  Action Variance: {metrics_run1['action_variance']:.6f}")
print(f"  Episodes to Convergence: {metrics_run1['episodes_to_convergence']}")
