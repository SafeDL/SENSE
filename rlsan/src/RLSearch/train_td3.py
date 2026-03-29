"""
train_td3.py - TD3专用训练脚本

使用离策 Twin Delayed Deep Deterministic Policy Gradient (TD3) 算法驱动小生境 PSO 搜索。
可以直接与 PPO, SAC 和 DDPG 的训练曲线对比。

使用方法：
    python train_td3.py --num_episodes 1000 --batch_size 256
"""

import torch
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
from scipy.io import savemat

# 本地导入
from rl_core.td3_agent import TD3Agent
from optimizer.niche_pso import StateAwareNichePSO, get_state_aware_state_dim
from optimizer.reward import SafetyRewardCalculator
from utils import set_seed, load_surrogate_model, get_dummy_surrogate_model, compute_rl_metrics

try:
    from envs.ad_scenario_env import ADScenarioEnv
    import gpytorch
except ImportError:
    ADScenarioEnv = None


def train_one_episode(pso: StateAwareNichePSO,
                      env,
                      agent: TD3Agent,
                      iterations_per_episode: int = 100) -> dict:
    """训练单个episode (TD3 Off-Policy)"""
    
    episode_rewards = []
    episode_actions = []
    
    episode_critic_losses = []
    episode_actor_losses = []
    
    agent.reset_noise()
    
    for i in range(iterations_per_episode):
        # PSO迭代 (执行一步)
        rewards = pso.run_one_iteration()
        episode_rewards.extend(rewards)
        
        # 统计连续动作参数
        for g in pso.subgroups:
            if 'last_action' in g:
                act = g['last_action']
                if isinstance(act, np.ndarray):
                    episode_actions.append(act.copy())
                elif isinstance(act, (list, tuple)):
                    episode_actions.append(np.array(act))
                    
        # 实时刷新 Buffer
        if hasattr(pso, 'flush_experience_to_agent'):
            pso.flush_experience_to_agent(done_all=False)
            
        # 判断是否在这一步产生了新的转换样本 (决策步)
        is_decision_step = (pso.iteration_count - 1) % pso.action_interval == 0
            
        # TD3 更新 (Off-Policy)
        if agent.buffer.size > agent.batch_size and is_decision_step:
            update_info = agent.update(updates=pso.num_subgroups)
            if update_info:
                episode_critic_losses.append(update_info.get('critic_loss', 0.0))
                # TD3 delayed policy updates means sometimes actor_loss is 0 from the returned dict if policy wasn't updated
                act_loss = update_info.get('actor_loss', 0.0)
                if act_loss != 0.0:
                    episode_actor_losses.append(act_loss)
            
    # Episode 结束，强制刷新剩余 Buffer 并标记 Done
    if hasattr(pso, 'flush_experience_to_agent'):
        pso.flush_experience_to_agent(done_all=True)
        
    if agent.buffer.size > agent.batch_size:
        update_info = agent.update(updates=pso.num_subgroups)
        if update_info:
             act_loss = update_info.get('actor_loss', 0.0)
             if act_loss != 0.0:
                  episode_actor_losses.append(act_loss)

    # 获取最优适应度
    best_fitness = min(g['global_best_fitness'] for g in pso.subgroups)
    
    # 统计危险场景
    danger_threshold = -0.3
    danger_found = best_fitness < danger_threshold
    num_danger_groups = sum(1 for g in pso.subgroups if g['global_best_fitness'] < danger_threshold)
    
    stats = pso.get_training_stats()
    reward_stats = pso.reward_calculator.get_stats()
    
    # 动作均值
    if episode_actions:
        action_means = np.mean(episode_actions, axis=0)
    else:
        action_means = np.zeros(4)
        
    return {
        'mean_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
        'best_fitness': best_fitness,
        'num_niches': stats['num_niches'],
        'coverage_rate': stats['coverage_rate'],
        'action_means': action_means,
        'episode_actions': episode_actions,
        'num_danger_groups': num_danger_groups,
        'danger_found': danger_found,
        'milestones_achieved': reward_stats.get('milestones_achieved', 0),
        'best_fitness_ever': reward_stats.get('best_fitness', float('inf')),
        'buffer_size': agent.buffer.size,
        'critic_loss': np.mean(episode_critic_losses) if episode_critic_losses else 0.0,
        'actor_loss': np.mean(episode_actor_losses) if episode_actor_losses else 0.0,
    }


def plot_training_curves(rewards: list, actor_losses: list, critic_losses: list,
                         action_history: list, save_path: str):
    """绘制TD3训练曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    window = 10
    
    # 1. 奖励曲线
    if len(rewards) > 0:
        axes[0, 0].plot(rewards, alpha=0.3, color='blue', label='Raw')
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            padding = np.full(window-1, np.nan)
            smoothed_plot = np.concatenate([padding, smoothed])
            axes[0, 0].plot(smoothed_plot, color='darkblue', linewidth=2, label=f'MA({window})')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Mean Reward')
    axes[0, 0].set_title('Training Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    
    # 2. Actor Loss
    valid = [l for l in actor_losses if l is not None and not np.isnan(l)]
    if valid:
        axes[0, 1].plot(valid, color='red', alpha=0.5)
        if len(valid) >= window:
            smoothed = np.convolve(valid, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(range(window-1, len(valid)), smoothed, color='red', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Actor Loss')
    axes[0, 1].set_title('Actor Loss (-Q1)')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    
    # 3. 占位 (TD3 没有 Alpha)
    axes[0, 2].text(0.5, 0.5, 'TD3 (No Entropy)', ha='center', va='center')
    axes[0, 2].axis('off')
    
    # 4. 连续动作参数变化
    if action_history:
        action_array = np.array(action_history)
        param_names = ['w', 'c1', 'c2', 'vel_scale']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, (name, color) in enumerate(zip(param_names, colors)):
            if i < action_array.shape[1]:
                axes[1, 0].plot(action_array[:, i], label=name, alpha=0.7, color=color)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Parameter Value')
        axes[1, 0].set_title('Continuous Action Parameters')
        axes[1, 0].legend()
        axes[1, 0].grid(True, linestyle='--', alpha=0.5)
    else:
        axes[1, 0].text(0.5, 0.5, 'No action data', ha='center', va='center',
                       transform=axes[1, 0].transAxes)
    
    # 5. Critic Loss
    valid = [l for l in critic_losses if l is not None and not np.isnan(l)]
    if valid:
        axes[1, 1].plot(valid, color='green', alpha=0.5)
        if len(valid) >= window:
            smoothed = np.convolve(valid, np.ones(window)/window, mode='valid')
            axes[1, 1].plot(range(window-1, len(valid)), smoothed, color='green', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Critic Loss')
    axes[1, 1].set_title('Critic Loss (Q1+Q2)')
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)
    
    # 6. 奖励分布直方图
    if len(rewards) > 10:
        last_rewards = rewards[-50:]
        axes[1, 2].hist(last_rewards, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        axes[1, 2].axvline(np.mean(last_rewards), color='red', linestyle='--', label=f'Mean: {np.mean(last_rewards):.3f}')
        axes[1, 2].set_xlabel('Reward')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Reward Distribution (Last 50 Episodes)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='TD3 RL-PSO Training')
    
    parser.add_argument('--num_episodes', type=int, default=500)
    parser.add_argument('--iterations_per_episode', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_particles', type=int, default=500)
    parser.add_argument('--num_subgroups', type=int, default=10)
    
    parser.add_argument('--dim', type=int, default=3)
    parser.add_argument('--bounds', type=float, nargs=2, default=[-1, 1])
    
    # TD3 Params
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.7, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='Target network update rate')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--exploration_noise', type=float, default=0.2, help='Gaussian exploration noise std')
    parser.add_argument('--disable_ou_noise', action='store_true', help='Disable OU process and use pure Gaussian noise')
    parser.add_argument('--policy_noise', type=float, default=0.3, help='Noise added to target policy')
    parser.add_argument('--noise_clip', type=float, default=0.6, help='Range to clip target policy noise')
    parser.add_argument('--policy_delay', type=int, default=3, help='Frequency of delayed policy updates')

    # ADS params
    parser.add_argument('--model_path', type=str, default='../../results/s1exp/surrogate_model.pkl')
    parser.add_argument('--danger_threshold', type=float, default=-0.3, help='Threshold for niche discovery')
    
    # 策略执行区间和动作平滑
    parser.add_argument('--action_interval', type=int, default=10)
    parser.add_argument('--action_smoothing', type=float, default=0.6)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TD3 RL-PSO Training (Aligned Version)")
    print("=" * 60)
    print("Alignment Features:")
    print("  ✓ Network hidden dim: 128x128x64")
    print("  ✓ Discount Factor (gamma): 0.7")
    print("  ✓ Particles config: Match PPO/SAC/DDPG defaults")
    print("=" * 60)
    print(f"✅ 使用TD3连续动作算法 | Target Noise: {args.policy_noise} | Delay: {args.policy_delay} | OU Noise: {not args.disable_ou_noise}")
    print(f"   action_interval: {args.action_interval}")
    print(f"   action_smoothing: {args.action_smoothing}")
    print("=" * 60)
    
    # 路径设置
    exp_name = f"TD3_ads_p{args.num_particles}_g{args.num_subgroups}_{args.seed}"
    result_dir = os.path.join("results", exp_name)
    os.makedirs(result_dir, exist_ok=True)
    
    set_seed(args.seed)
    
    # 环境创建 (仅支持 ADS 自动驾驶场景)
    if ADScenarioEnv is None:
        print("Error: ADS mode requires 'gpytorch'")
        exit(1)
        
    print(f"\n🚗 ADS测试场景训练模式")
    try:
        gp_model, gp_likelihood = load_surrogate_model(args.model_path)
        print("✅ Surrogate model loaded")
    except Exception as e:
        print(f"❌ Error: {e}, using dummy model")
        gp_model, gp_likelihood = get_dummy_surrogate_model(args.dim)
    
    env = ADScenarioEnv(gp_model=gp_model, gp_likelihood=gp_likelihood,
                       dim=args.dim, bounds=tuple(args.bounds), runner=None)
    
    state_dim = get_state_aware_state_dim()
    
    # Agent初始化
    agent = TD3Agent(
        state_dim=state_dim,
        action_dim=4,
        lr=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        batch_size=args.batch_size,
        hidden_dims=[args.hidden_dim, args.hidden_dim, 64],
        exploration_noise=args.exploration_noise,
        use_ou_noise=not args.disable_ou_noise,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_delay=args.policy_delay
    )
    
    print(f"Agent initialized: TD3")
    
    pso = StateAwareNichePSO(
        env=env, agent=agent,
        num_particles=args.num_particles,
        num_subgroups=args.num_subgroups,
        max_iterations=args.iterations_per_episode,
        enable_restart=True,
        restart_patience=5,
        danger_threshold=args.danger_threshold,
        task_type='ads',
        action_interval=args.action_interval,
        action_smoothing_factor=args.action_smoothing,
        reward_calculator_class=SafetyRewardCalculator,
        use_subgroup_features=True
    )

    all_rewards, all_actor_losses, all_critic_losses = [], [], []
    all_action_means = []
    all_best_fitness = []
    
    start_time = time.time()
    for episode in range(args.num_episodes):
        func_name = env.get_current_function_name() if hasattr(env, 'get_current_function_name') else 'ads'

        if episode == 0:
            pso.init_subgroups(reset_trackers=True)
            
        metrics = train_one_episode(pso, env, agent, args.iterations_per_episode)
        
        all_rewards.append(metrics['mean_reward'])
        if metrics['actor_loss'] != 0.0:
            all_actor_losses.append(metrics['actor_loss'])
        all_critic_losses.append(metrics['critic_loss'])
        all_action_means.append(metrics['action_means'])
        all_best_fitness.append(metrics['best_fitness'])
        
        window = min(20, len(all_rewards))
        avg_reward = np.mean(all_rewards[-window:])
        elapsed = time.time() - start_time
        speed = (episode + 1) / elapsed
        remaining = (args.num_episodes - episode - 1) / speed if speed > 0 else 0
        
        am = metrics['action_means']
        action_str = f"w:{am[0]:.2f} c1:{am[1]:.2f} c2:{am[2]:.2f} vs:{am[3]:.2f}"
        
        danger_flag = "🚨" if metrics.get('danger_found', False) else "  "
        danger_count = metrics.get('num_danger_groups', 0)
        best_fit = metrics['best_fitness']
        
        print(f"Ep {episode:3d}/{args.num_episodes} | {func_name:<15} | "
              f"R: {metrics['mean_reward']:.3f} | Avg: {avg_reward:.3f} | "
              f"Fit: {best_fit:.4f} {danger_flag}({danger_count}) | "
              f"{action_str} | ETA: {remaining/60:.1f}m")
            
        if episode > 0 and episode % 50 == 0:
             agent.save(os.path.join(result_dir, f"td3_model_ep{episode}.pth"))
             
    plot_path = os.path.join(result_dir, 'td3_training_curves.png')
    plot_training_curves(all_rewards, all_actor_losses, all_critic_losses,
                         all_action_means, plot_path)
    
    agent.save(os.path.join(result_dir, f"td3_model_final.pth"))
    
    # 导出训练数据为 .mat 格式（供 MATLAB 对比可视化）
    action_array = np.array(all_action_means)
    mat_path = os.path.join(result_dir, 'td3_training_data.mat')
    savemat(mat_path, {
        'algorithm': 'TD3',
        'episodes': np.arange(1, len(all_rewards) + 1),
        'mean_rewards': np.array(all_rewards),
        'best_fitness': np.array(all_best_fitness),
        'actor_losses': np.array(all_actor_losses),
        'critic_losses': np.array(all_critic_losses),
        'action_w': action_array[:, 0] if len(action_array) > 0 else np.array([]),
        'action_c1': action_array[:, 1] if len(action_array) > 0 else np.array([]),
        'action_c2': action_array[:, 2] if len(action_array) > 0 else np.array([]),
        'action_vs': action_array[:, 3] if len(action_array) > 0 else np.array([]),
    })
    print(f"Training data saved to: {mat_path}")
    
    print("\n" + "=" * 60)
    print("Reward Calculator Statistics:")
    pso.reward_calculator.print_stats()
    
    # 评测并输出比较指标
    compute_rl_metrics(all_rewards, all_action_means, pso)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Time: {(time.time() - start_time)/60:.1f}m | Final Avg Reward: {np.mean(all_rewards[-20:]):.4f}")
    print(f"Curves: {plot_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
