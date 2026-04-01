import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
data_multi = sio.loadmat('ddpg_multi_run_data.mat')
data_run1 = sio.loadmat('ddpg_run_1.mat')

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 平均奖励曲线
episodes = data_multi['episodes'].flatten()
mean_rewards_multi = data_multi['mean_rewards'].flatten()
std_rewards_multi = data_multi['std_rewards'].flatten()

axes[0, 0].plot(episodes, mean_rewards_multi, 'b-', linewidth=2, label='Multi-Run Mean')
axes[0, 0].fill_between(episodes, mean_rewards_multi - std_rewards_multi,
                         mean_rewards_multi + std_rewards_multi, alpha=0.3)
axes[0, 0].set_xlabel('Episodes')
axes[0, 0].set_ylabel('Mean Rewards')
axes[0, 0].set_title('DDPG 平均奖励')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# 2. 单次运行奖励
mean_rewards_run1 = data_run1['mean_rewards'].flatten()
axes[0, 1].plot(episodes, mean_rewards_run1, 'r-', linewidth=2, label='Run 1')
axes[0, 1].set_xlabel('Episodes')
axes[0, 1].set_ylabel('Mean Rewards')
axes[0, 1].set_title('DDPG Run 1 奖励')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# 3. Actor Loss
actor_losses_multi = data_multi['all_actor_losses']
axes[1, 0].plot(episodes, np.mean(actor_losses_multi, axis=0), 'g-', linewidth=2)
axes[1, 0].set_xlabel('Episodes')
axes[1, 0].set_ylabel('Actor Loss')
axes[1, 0].set_title('DDPG Actor Loss')
axes[1, 0].grid(True, alpha=0.3)

# 4. Critic Loss
critic_losses_multi = data_multi['all_critic_losses']
axes[1, 1].plot(episodes, np.mean(critic_losses_multi, axis=0), 'm-', linewidth=2)
axes[1, 1].set_xlabel('Episodes')
axes[1, 1].set_ylabel('Critic Loss')
axes[1, 1].set_title('DDPG Critic Loss')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ddpg_results.png', dpi=150, bbox_inches='tight')
print("图表已保存为: ddpg_results.png")
plt.close()

# 打印指标总结
print("\n" + "="*50)
print("DDPG 结果总结")
print("="*50)
print(f"Asymptotic Return (Multi-Run): {np.mean(mean_rewards_multi[-100:]):.4f}")
print(f"Asymptotic Return (Run 1): {np.mean(mean_rewards_run1[-100:]):.4f}")
print(f"Episodes to Convergence (Multi-Run): {np.where(mean_rewards_multi >= np.max(mean_rewards_multi)*0.95)[0][0]}")
print(f"Episodes to Convergence (Run 1): {np.where(mean_rewards_run1 >= np.max(mean_rewards_run1)*0.95)[0][0]}")
