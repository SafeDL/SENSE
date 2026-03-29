"""
train_trpo.py - TRPO专用训练脚本

基于 train_ppo.py 改造的 On-Policy 测试基准。
用于运行 TRPOAgent。

使用方法：
    python train_trpo.py --num_episodes 200
    python train_trpo.py --mode ads_test --model_path ../../results/s1exp/surrogate_model.pkl
"""

import argparse
import numpy as np
import torch
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import savemat

# 本地导入
from rl_core.trpo_agent import TRPOAgent
from optimizer.niche_pso import StateAwareNichePSO, get_state_aware_state_dim
from optimizer.reward import SafetyRewardCalculator
from utils import set_seed, load_surrogate_model, get_dummy_surrogate_model, compute_rl_metrics






def train_one_episode(pso: StateAwareNichePSO,
                      env,
                      agent: TRPOAgent,
                      iterations_per_episode: int = 100) -> dict:
    """训练单个episode（连续动作）"""
    
    episode_rewards = []
    episode_actions = []
    
    for i in range(iterations_per_episode):
        # PSO迭代
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
    
    # 刷新所有并行的轨迹缓存，并标记为Done
    if hasattr(pso, 'flush_experience_to_agent'):
        pso.flush_experience_to_agent(done_all=True)
    
    # Episode结束时TRPO更新
    update_info = agent.update(next_value=0.0) if hasattr(agent, 'update') else {
        'policy_loss': 0, 'value_loss': 0, 'kl_divergence': 0
    }
    
    # 获取最优适应度
    best_fitness = min(g['global_best_fitness'] for g in pso.subgroups)
    
    # 检测是否发现危险场景
    danger_threshold = getattr(pso, 'danger_threshold', -0.3)
    danger_found = best_fitness < danger_threshold
    num_danger_groups = sum(1 for g in pso.subgroups if g['global_best_fitness'] < danger_threshold)
    
    # 获取统计信息
    stats = pso.get_training_stats()
    
    # 获取奖励计算器统计
    reward_stats = pso.reward_calculator.get_stats()
    
    # 计算连续动作参数的均值
    if episode_actions:
        action_means = np.mean(episode_actions, axis=0)
    else:
        action_means = np.zeros(4)
    
    return {
        'mean_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
        'policy_loss': update_info.get('policy_loss', 0.0),
        'value_loss': update_info.get('value_loss', 0.0),
        'kl_divergence': update_info.get('kl_divergence', 0.0),
        'best_fitness': best_fitness,
        'num_niches': stats['num_niches'],
        'coverage_rate': stats['coverage_rate'],
        'action_means': action_means,
        'episode_actions': episode_actions,
        'milestones_achieved': reward_stats.get('milestones_achieved', 0),
        'best_fitness_ever': reward_stats.get('best_fitness', float('inf')),
        'danger_found': danger_found,
        'num_danger_groups': num_danger_groups,
    }


def plot_training_curves(rewards: list, policy_losses: list, value_losses: list,
                         kl_divs: list, action_history: list,
                         save_path: str):
    """绘制TRPO训练曲线"""
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
    axes[0, 0].set_title('Training Reward (Fixed Scale)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    
    # 2. Policy Loss
    valid = [l for l in policy_losses if l is not None and not np.isnan(l)]
    if valid:
        axes[0, 1].plot(valid, color='red', alpha=0.5)
        if len(valid) >= window:
            smoothed = np.convolve(valid, np.ones(window)/window, mode='valid')
            axes[0, 1].plot(range(window-1, len(valid)), smoothed, color='red', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Policy Loss')
    axes[0, 1].set_title('Policy Loss')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    
    # 3. KL Divergence
    ax3 = axes[0, 2]
    valid = [k for k in kl_divs if k is not None and not np.isnan(k)]
    if valid:
        ax3.plot(valid, color='purple', alpha=0.5, label='KL Div')
        if len(valid) >= window:
            smoothed = np.convolve(valid, np.ones(window)/window, mode='valid')
            ax3.plot(range(window-1, len(valid)), smoothed, color='purple', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('KL Divergence', color='purple')
    ax3.tick_params(axis='y', labelcolor='purple')
    
    ax3.set_title('KL Divergence (max_kl strict constraint)')
    ax3.grid(True, linestyle='--', alpha=0.5)
    
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
    
    # 5. Value Loss
    valid = [l for l in value_losses if l is not None and not np.isnan(l)]
    if valid:
        axes[1, 1].plot(valid, color='green', alpha=0.5)
        if len(valid) >= window:
            smoothed = np.convolve(valid, np.ones(window)/window, mode='valid')
            axes[1, 1].plot(range(window-1, len(valid)), smoothed, color='green', linewidth=2)
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Value Loss')
    axes[1, 1].set_title('Value Loss')
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)
    
    # 6. 奖励分布直方图（最后50个episode）
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
    parser = argparse.ArgumentParser(description='TRPO RL-PSO Training')
    
    # 训练参数
    parser.add_argument('--num_episodes', type=int, default=500)
    parser.add_argument('--iterations_per_episode', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    
    # PSO参数
    parser.add_argument('--num_particles', type=int, default=500)
    parser.add_argument('--num_subgroups', type=int, default=10)
    parser.add_argument('--dim', type=int, default=3)
    parser.add_argument('--bounds', type=float, nargs=2, default=[-1, 1])
    
    # TRPO参数
    parser.add_argument('--value_lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--max_kl', type=float, default=0.01)
    parser.add_argument('--cg_damping', type=float, default=0.1)
    parser.add_argument('--cg_iters', type=int, default=10)
    parser.add_argument('--value_epochs', type=int, default=10)
    parser.add_argument('--min_batch_size', type=int, default=1000)

    parser.add_argument('--danger_threshold', type=float, default=-0.3, help='Threshold for niche discovery')
    
    # ADS params
    parser.add_argument('--model_path', type=str, default='../../results/s1exp/surrogate_model.pkl')
    
    # 策略执行区间
    parser.add_argument('--action_interval', type=int, default=10)
    parser.add_argument('--action_smoothing', type=float, default=0.6)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TRPO RL-PSO Training Base Script")
    print("=" * 60)
    print("Key Improvements:")
    print("  ✓ Strict KL Divergence Constraints")
    print("  ✓ Trust Region Policy Optimization via Fisher Vector Products")
    print("=" * 60)
    
    # 获取PPO专用状态维度（28维，包含子种群感知特征）
    state_dim = get_state_aware_state_dim()
    action_dim = 4
    
    print(f"✅ 使用 TRPO 连续动作算法")
    print(f"   State dim: {state_dim} (25 base + 3 subgroup-aware)")
    print(f"   Action dim: {action_dim} (w, c1, c2, velocity_scale)")
    print(f"   Max KL constraint: {args.max_kl}")
    print(f"   Episodes: {args.num_episodes}, Iterations: {args.iterations_per_episode}")
    print("=" * 60)
    
    # 路径设置
    exp_name = f"TRPO_ads_p{args.num_particles}_g{args.num_subgroups}_{args.seed}"
    result_dir = os.path.join("results", exp_name)
    os.makedirs(result_dir, exist_ok=True)
    
    set_seed(args.seed)
    
    # 创建环境 (仅支持 ADS 自动驾驶场景)
    try:
        from envs.ad_scenario_env import ADScenarioEnv
        import gpytorch
    except ImportError:
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
    
    # 创建连续动作TRPO Agent
    agent = TRPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        max_kl=args.max_kl,
        cg_damping=args.cg_damping,
        cg_iters=args.cg_iters,
        value_lr=args.value_lr,
        value_epochs=args.value_epochs,
        min_batch_size=args.min_batch_size,
        hidden_dims=[128, 128, 64]
    )
    print(f"   Agent: TRPOAgent")
    
    # 创建TRPO专用PSO (与PPO逻辑相同)
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
    
    # 训练
    all_rewards, all_policy_losses, all_value_losses, all_kl_divs = [], [], [], []
    all_action_means = []
    all_best_fitness = []
    func_stats = {}
    
    start_time = time.time()
    
    for episode in range(args.num_episodes):
        func_name = env.get_current_function_name()
        
        if episode == 0:
            pso.init_subgroups(reset_trackers=True)
        
        result = train_one_episode(pso, env, agent, args.iterations_per_episode)
        
        all_rewards.append(result['mean_reward'])
        all_policy_losses.append(result['policy_loss'])
        all_value_losses.append(result['value_loss'])
        all_kl_divs.append(result['kl_divergence'])
        all_action_means.append(result['action_means'])
        all_best_fitness.append(result['best_fitness'])
        
        # 统计
        if func_name not in func_stats:
            func_stats[func_name] = {'count': 0, 'total_reward': 0, 'best_fitness': float('inf')}
        func_stats[func_name]['count'] += 1
        func_stats[func_name]['total_reward'] += result['mean_reward']
        func_stats[func_name]['best_fitness'] = min(func_stats[func_name]['best_fitness'], result['best_fitness'])
        
        # 进度
        window = min(20, len(all_rewards))
        avg_reward = np.mean(all_rewards[-window:])
        elapsed = time.time() - start_time
        speed = (episode + 1) / elapsed
        remaining = (args.num_episodes - episode - 1) / speed if speed > 0 else 0
        
        # 动作参数显示
        am = result['action_means']
        action_str = f"w:{am[0]:.2f} c1:{am[1]:.2f} c2:{am[2]:.2f} vs:{am[3]:.2f}"
        
        # 危险场景标志
        danger_flag = "🚨" if result.get('danger_found', False) else "  "
        danger_count = result.get('num_danger_groups', 0)
        best_fit = result['best_fitness']
        
        print(f"Ep {episode:3d}/{args.num_episodes} | {func_name:<15} | "
              f"R: {result['mean_reward']:.3f} | Avg: {avg_reward:.3f} | "
              f"Fit: {best_fit:.4f} {danger_flag}({danger_count}) | "
              f"KL: {result['kl_divergence']:.4f} | "
              f"{action_str} | ETA: {remaining/60:.1f}m")
              
        if episode > 0 and episode % 50 == 0:
             agent.save(os.path.join(result_dir, f"trpo_model_ep{episode}.pth"))
    
    # 保存
    save_path = os.path.join(result_dir, 'trpo_model_final.pth')
    plot_path = os.path.join(result_dir, 'trpo_training_curves.png')
    
    agent.save(save_path)
    plot_training_curves(all_rewards, all_policy_losses, all_value_losses,
                         all_kl_divs, all_action_means, plot_path)
    
    # 导出训练数据为 .mat 格式（供 MATLAB 对比可视化）
    action_array = np.array(all_action_means)
    mat_path = os.path.join(result_dir, 'trpo_training_data.mat')
    savemat(mat_path, {
        'algorithm': 'TRPO',
        'episodes': np.arange(1, len(all_rewards) + 1),
        'mean_rewards': np.array(all_rewards),
        'best_fitness': np.array(all_best_fitness),
        'policy_losses': np.array(all_policy_losses),
        'value_losses': np.array(all_value_losses),
        'kl_divergences': np.array(all_kl_divs),
        'action_w': action_array[:, 0] if len(action_array) > 0 else np.array([]),
        'action_c1': action_array[:, 1] if len(action_array) > 0 else np.array([]),
        'action_c2': action_array[:, 2] if len(action_array) > 0 else np.array([]),
        'action_vs': action_array[:, 3] if len(action_array) > 0 else np.array([]),
    })
    print(f"Training data saved to: {mat_path}")
    
    # 打印奖励计算器统计
    print("\n" + "=" * 60)
    print("Reward Calculator Statistics:")
    pso.reward_calculator.print_stats()
    
    # 评测并输出比较指标
    compute_rl_metrics(all_rewards, all_action_means, pso)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Time: {(time.time() - start_time)/60:.1f}m | Final Avg Reward: {np.mean(all_rewards[-20:]):.4f}")
    print(f"Model: {save_path}")
    print(f"Curves: {plot_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()