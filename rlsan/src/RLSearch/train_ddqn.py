"""
train_ddqn.py - 状态感知RL-PSO训练脚本

使用改进版组件：
- StateAwareNichePSO: 状态感知小生境PSO
- 8个简化动作
- 20维状态特征
- 聚类多样性奖励

使用方法：
    python train_ddqn.py --num_episodes 100
"""

import argparse
import numpy as np
import torch
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 本地导入
from rl_core.dqn_agent import DoubleDQNAgent
from optimizer.niche_pso import StateAwareNichePSO, get_state_aware_state_dim, get_state_aware_action_dim
from optimizer.reward import SafetyRewardCalculator
from utils import set_seed, load_surrogate_model, get_dummy_surrogate_model, compute_rl_metrics


def train_one_episode(pso: StateAwareNichePSO,
                      env,
                      agent: DoubleDQNAgent,
                      iterations_per_episode: int = 100) -> dict:
    """训练单个episode"""
    
    episode_rewards = []
    episode_losses = []
    episode_q_values = []
    episode_probs = [] # 记录所有步的所有动作概率 (n_steps, n_subgroups, n_actions)
    action_counts = np.zeros(6)  # 6个动作的计数
    
    for i in range(iterations_per_episode):
        # PSO迭代
        rewards = pso.run_one_iteration()
        episode_rewards.extend(rewards)
        
        # 统计动作分布 & 收集动作概率
        current_step_probs = []
        for g in pso.subgroups:
            if 'last_action' in g:
                act = g['last_action']
                action_counts[act] += 1
                
                # 获取概率分布 (优先使用 Softmax 概率)
                if 'last_probs' in g and g['last_probs'] is not None:
                    prob = g['last_probs']
                else:
                    # Fallback to One-Hot
                    prob = np.zeros(6)
                    prob[act] = 1.0
                current_step_probs.append(prob)
        
        episode_probs.append(current_step_probs)
        
        # 实时刷新 Buffer
        if hasattr(pso, 'flush_experience_to_agent'):
            pso.flush_experience_to_agent(done_all=False)
            
        # 判断是否在这一步产生了新的转换样本 (决策步)
        is_decision_step = (pso.iteration_count - 1) % pso.action_interval == 0
        
        # RL Agent更新 (仅在新样本产生后更新，防止过拟合)
        if is_decision_step:
            # 每次决策步产生 num_subgroups 个新样本，更新与其匹配的次数 (保持 UTD)
            for _ in range(pso.num_subgroups):
                loss, q_val = agent.update_q_values()
                if loss is not None:
                    episode_losses.append(loss)
                if q_val is not None:
                    episode_q_values.append(q_val)
                    
            # 软更新目标网络
            agent.soft_update_target_network(tau=0.01)
    
    # epsilon衰减
    agent.decay_epsilon()
    
    # Episode 结束，强制刷新剩余 Buffer 并标记 Done
    if hasattr(pso, 'flush_experience_to_agent'):
        pso.flush_experience_to_agent(done_all=True)
        
        # 补上一轮更新
        for _ in range(pso.num_subgroups):
            loss, q_val = agent.update_q_values()
            if loss is not None:
                episode_losses.append(loss)
            if q_val is not None:
                episode_q_values.append(q_val)
    
    # 获取最优适应度
    best_fitness = min(g['global_best_fitness'] for g in pso.subgroups)
    
    # 检测是否发现危险场景
    danger_threshold = getattr(pso, 'danger_threshold', -0.3)
    danger_found = best_fitness < danger_threshold
    num_danger_groups = sum(1 for g in pso.subgroups if g['global_best_fitness'] < danger_threshold)
    
    # 获取统计信息
    stats = pso.get_training_stats()
    
    # 计算动作分布（归一化）
    action_dist = action_counts / (action_counts.sum() + 1e-8)
    
    return {
        'mean_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
        'mean_loss': np.mean(episode_losses) if episode_losses else 0.0,
        'mean_q_value': np.mean(episode_q_values) if episode_q_values else 0.0,
        'best_fitness': best_fitness,
        'epsilon': agent.epsilon,
        'num_niches': stats['num_niches'],
        'coverage_rate': stats['coverage_rate'],
        'action_dist': action_dist,
        'episode_probs': episode_probs,
        'danger_found': danger_found,
        'num_danger_groups': num_danger_groups,
    }


def plot_training_curves(rewards: list, losses: list, q_values: list,
                         action_dists: list, probs_history: list, save_path: str):
    """绘制训练曲线
    
    Args:
        rewards: 每个episode的平均奖励
        losses: 每个episode的平均损失
        q_values: 每个episode的平均Q值
        action_dists: 每个episode的动作分布 (n_episodes, 6)
        probs_history: 所有步的详情动作概率历史 (list of list of prob arrays)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    window = 10
    
    # 1. 奖励曲线
    if len(rewards) > 0:
        axes[0, 0].plot(rewards, color='blue', label='Raw', alpha=0.3)
        
        # Smoothed curve
        if len(rewards) >= window:
            weights = np.ones(window) / window
            smoothed = np.convolve(rewards, weights, mode='valid')
            # padding for alignment
            padding = np.full(window-1, np.nan)
            smoothed_plot = np.concatenate([padding, smoothed])
            axes[0, 0].plot(smoothed_plot, color='darkblue', label=f'Avg({window})', linewidth=2)
            
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Mean Reward')
    axes[0, 0].set_title('Training Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)
    
    # 2. 损失曲线
    valid_losses = [l for l in losses if l is not None and l > 0]
    if valid_losses:
        axes[0, 1].plot(valid_losses, color='red', alpha=0.8)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Training Loss (Raw)')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    
    # 3. Q值曲线
    valid_q = [q for q in q_values if q is not None]
    if valid_q:
        axes[1, 0].plot(valid_q, color='green', alpha=0.8)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Average Q-Value')
    axes[1, 0].set_title('Q-Value Estimation (Raw)')
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)
    
    # 4. 动作分布热力图 (Time-based History)
    if probs_history:
        # probs_history is list of steps. Each step is list of subgroup arrays.
        # Step 1: Compute average probability vector for each time step
        # shape: (total_steps, 6)
        step_avg_probs = []
        discrete_actions = [] # 用于调试打印
        
        for step_data in probs_history:
             if not step_data: continue
             # step_data: list of arrays
             avg_prob = np.mean(np.array(step_data), axis=0)
             step_avg_probs.append(avg_prob)
             discrete_actions.append(np.argmax(avg_prob)) # 记录最可能的动作
        
        step_avg_probs = np.array(step_avg_probs)
        total_steps = len(step_avg_probs)
        
        print(f"DEBUG: Total Steps recorded: {total_steps}")
        if total_steps > 0:
            print(f"DEBUG: Sample Avg Probs (Step 0): {step_avg_probs[0]}")
            print(f"DEBUG: Sample Actions (Argmax): {discrete_actions[:50]}") # Print first 50 actions
        
        # Binning (fixed 100 bins for clarity)
        num_bins = min(100, total_steps)
        if num_bins == 0: num_bins = 1
        bin_size = max(1, total_steps // num_bins)
        
        heatmap_data = np.zeros((6, num_bins))
        
        for i in range(num_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < num_bins - 1 else total_steps
            if start_idx >= total_steps: break
            
            segment = step_avg_probs[start_idx:end_idx] # (segment_len, 6)
            
            # Average over the segment (time window)
            # 这里的含义是：在这个时间段内，平均的策略分布是什么
            avg_segment_prob = np.mean(segment, axis=0) # (6,)
            heatmap_data[:, i] = avg_segment_prob
            
        print(f"DEBUG: Heatmap max value: {np.max(heatmap_data)}")
        print(f"DEBUG: Heatmap min value: {np.min(heatmap_data)}")
        
        im = axes[1, 1].imshow(heatmap_data, aspect='auto', cmap='viridis', 
                               interpolation='nearest', vmin=0.0, vmax=1.0,
                               origin='lower', extent=[0, total_steps, -0.5, 5.5])

        axes[1, 1].set_xlabel(f'Total Steps (x{bin_size})')
        axes[1, 1].set_ylabel('Action Index')
        axes[1, 1].set_title('Policy Distribution Over Time (Softmax)')
        axes[1, 1].set_yticks(range(6))
        axes[1, 1].set_yticklabels(['WideScout', 'LocalAttack', 'Balanced', 
                                     'Escape', 'VelBoost', 'VelDampen'])
        plt.colorbar(im, ax=axes[1, 1], label='Probability')
    elif action_dists:
        # Fallback to episode-based
        action_matrix = np.array(action_dists).T
        im = axes[1, 1].imshow(action_matrix, aspect='auto', cmap='YlOrRd',
                               interpolation='nearest', vmin=0, vmax=1.0)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_title('Action Distribution (Episode-averaged)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='RL-PSO Training')
    
    # 训练参数
    parser.add_argument('--num_episodes', type=int, default=500)
    parser.add_argument('--iterations_per_episode', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    
    # PSO参数
    parser.add_argument('--num_particles', type=int, default=500)
    parser.add_argument('--num_subgroups', type=int, default=5)
    parser.add_argument('--dim', type=int, default=3)  # 与ADS部署一致 (surrogate model trained on 3 dims)
    parser.add_argument('--bounds', type=float, nargs=2, default=[-1, 1])  # 统一为[-1, 1]与ADS部署一致
    
    # DQN参数
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--buffer_size', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epsilon_start', type=float, default=.0)
    parser.add_argument('--epsilon_end', type=float, default=0.05)
    parser.add_argument('--epsilon_decay', type=float, default=0.995)
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='Software exploration temperature')
    
    # ADS params
    parser.add_argument('--danger_threshold', type=float, default=-0.3, 
                        help='Threshold for niche discovery')
    parser.add_argument('--model_path', type=str, default='../../results/s1exp/surrogate_model.pkl',
                        help='Path to surrogate model')

    # 策略执行区间与平滑 (Scheme A & B)
    parser.add_argument('--action_interval', type=int, default=10,
                        help='Number of steps to hold the same action (Frame Skipping)')
    parser.add_argument('--action_smoothing', type=float, default=0.6,
                        help='Smoothing factor (alpha) for parameter updates')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("State-Aware RL-PSO Training")
    print("=" * 60)
    
    # 获取维度
    state_dim = get_state_aware_state_dim()
    action_dim = get_state_aware_action_dim()
    
    print(f"✅ 使用状态感知框架")
    print(f"   State dim: {state_dim} ")
    print(f"   Action dim: {action_dim}")
    print(f"Episodes: {args.num_episodes}, Iterations: {args.iterations_per_episode}")
    print(f"Particles: {args.num_particles}, Subgroups: {args.num_subgroups}")
    print("=" * 60)
    
    # 路径设置
    exp_name = f"DDQN_ads_p{args.num_particles}_g{args.num_subgroups}_{args.seed}"
    result_dir = os.path.join("results", exp_name)
    os.makedirs(result_dir, exist_ok=True)
    
    set_seed(args.seed)
    
    # 环境创建 (仅支持 ADS 自动驾驶场景)
    try:
        from envs.ad_scenario_env import ADScenarioEnv
        import gpytorch
    except ImportError:
        print("Error: ADS mode requires 'gpytorch'. Please install it.")
        exit(1)
        
    print(f"\n🚗 ADS测试场景训练模式")
    print(f"   Model Path: {args.model_path}")
    
    # 加载真实代理模型
    try:
         gp_model, gp_likelihood = load_surrogate_model(args.model_path)
         print("✅ Surrogate model loaded successfully.")
    except Exception as e:
          print(f"❌ Error loading surrogate model: {e}")
          print("   Falling back to Dummy GP Model for demonstration ONLY.")
          gp_model, gp_likelihood = get_dummy_surrogate_model(args.dim)

    env = ADScenarioEnv(
        gp_model=gp_model,
        gp_likelihood=gp_likelihood,
        dim=args.dim,
        bounds=tuple(args.bounds),
        runner=None
    )
    print(f"   ADS Environment Initialized (Surrogate Only)")

    # 创建Agent（使用更小的网络因为状态/动作都更小）
    agent = DoubleDQNAgent(
        state_dim=state_dim,
        num_actions=action_dim,
        alpha=args.lr,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        epsilon=args.epsilon_start,
        min_epsilon=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        temperature=args.temperature,
        hidden_dims=[128, 128, 64]  # 网络需要3层隐藏层
    )
    
    # 创建PSO
    pso = StateAwareNichePSO(
        env=env,
        agent=agent,
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
    all_rewards = []
    all_losses = []
    all_q_values = []
    all_action_dists = []
    all_prob_history = [] # 全局动作概率历史
    func_stats = {}
    
    start_time = time.time()
    
    print(f"\n✅ ADS场景训练: 代理模型辅助环境")
    print(f"   目标: 在 {args.dim} 维参数空间中搜索高风险场景\n")
    
    for episode in range(args.num_episodes):
        func_name = env.get_current_function_name()
        
        # 训练（首轮完整初始化）
        if episode == 0:
            pso.init_subgroups(reset_trackers=True)
        
        result = train_one_episode(
            pso, env, agent, args.iterations_per_episode
        )
        
        all_rewards.append(result['mean_reward'])
        all_losses.append(result['mean_loss'])
        all_q_values.append(result['mean_q_value'])
        all_action_dists.append(result['action_dist'])
        all_prob_history.extend(result['episode_probs']) # 展平为步的列表
        
        # 更新函数统计
        if func_name not in func_stats:
            func_stats[func_name] = {'count': 0, 'total_reward': 0, 'best_fitness': float('inf')}
        func_stats[func_name]['count'] += 1
        func_stats[func_name]['total_reward'] += result['mean_reward']
        func_stats[func_name]['best_fitness'] = min(
            func_stats[func_name]['best_fitness'], result['best_fitness']
        )
        
        # 计算滑动平均
        window = min(20, len(all_rewards))
        avg_reward = np.mean(all_rewards[-window:])
        
        # 计算速度
        elapsed = time.time() - start_time
        speed = (episode + 1) / elapsed
        remaining = (args.num_episodes - episode - 1) / speed if speed > 0 else 0
        
        # 温度衰减 (每 Episode)
        agent.temperature = max(0.5, agent.temperature * args.epsilon_decay)
        
        # 显示进度
        progress = episode / args.num_episodes
        phase = "📘" if progress < 0.3 else ("📗" if progress < 0.7 else "📙")
            
        act_dist_pct = (result['action_dist'] * 100).astype(int)
        
        # 危险场景标志
        danger_flag = "🚨" if result.get('danger_found', False) else "  "
        danger_count = result.get('num_danger_groups', 0)
        
        print(f"{phase} Ep {episode:3d}/{args.num_episodes} | "
              f"Func: {func_name:<15} | "
              f"R: {result['mean_reward']:7.3f} | "
              f"Avg: {avg_reward:7.3f} | "
              f"Fit: {result['best_fitness']:.4f} {danger_flag}({danger_count}) | "
              f"Niches: {result['num_niches']:2d} | "
              f"T: {agent.temperature:.3f} | "
              f"Acts: {act_dist_pct}% | "
              f"ETA: {remaining/60:.1f}m")
              
        if episode > 0 and episode % 50 == 0:
             agent.save(os.path.join(result_dir, f"ddqn_model_ep{episode}.pth"))
    
    # 保存模型 (带类别标识)
    save_path = os.path.join(result_dir, 'ddqn_model_final.pth')
    plot_path = os.path.join(result_dir, 'ddqn_training_curves.png')
    
    agent.save(save_path)
    print(f"\nModel saved to: {save_path}")
    
    # 绘制曲线（新指标：Q-value 和 Action Distribution）
    plot_training_curves(all_rewards, all_losses, all_q_values, 
                         all_action_dists, all_prob_history, plot_path)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
    print(f"Final Avg Reward (last 20): {np.mean(all_rewards[-20:]):.4f}")
    
    print("\n📊 Function Training Statistics:")
    print("-" * 50)
    sorted_funcs = sorted(func_stats.items(), key=lambda x: x[1]['count'], reverse=True)
    for func_name, stats in sorted_funcs:
        avg_reward = stats['total_reward'] / stats['count']
        print(f"  {func_name:<20} | count: {stats['count']:3d} | avg_reward: {avg_reward:7.3f}")
    print("-" * 50)
    print(f"Total functions used: {len(func_stats)}")

    # 评测并输出比较指标
    compute_rl_metrics(all_rewards, all_action_dists, pso)

    print("=" * 60)


if __name__ == '__main__':
    main()
