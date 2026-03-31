"""
train_ddqn.py - Double DQN 训练脚本（多运行版）

使用改进版组件：
- StateAwareNichePSO: 状态感知小生境PSO
- 离散动作空间
- 状态感知特征
- 聚类多样性奖励

使用方法：
    # 单次运行
    python train_ddqn.py --num_episodes 100 --num_runs 1

    # 多次独立实验 (RL标准做法: 3~5次)
    python train_ddqn.py --num_episodes 100 --num_runs 5 --seeds 42 123 456 789 1024
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
    episode_entropies = []   # 每步策略熵
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
        
        # 计算当前步策略熵：直接调用 pso._get_state()获取子群组状态
        step_entropies = []
        for idx in range(pso.num_subgroups):
            try:
                state = pso._get_state(idx)
                ent = agent.compute_policy_entropy(state)
                step_entropies.append(ent)
            except Exception:
                pass
        if step_entropies:
            episode_entropies.append(np.mean(step_entropies))
        
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
    
    # 计算最后20步的平均策略分布熵
    window = 20
    if episode_entropies:
        avg_policy_entropy = float(np.mean(episode_entropies[-window:]))
    else:
        avg_policy_entropy = 0.0
    
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
        'avg_policy_entropy': avg_policy_entropy,   # 新增：最后20步平均策略熵
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


def run_single_experiment(args, run_idx: int, seed: int) -> dict:
    """运行一次完整的 DDQN 训练实验"""
    print(f"\n{'#' * 60}\n# Run {run_idx + 1} | Seed: {seed}\n{'#' * 60}")
    set_seed(seed)
    exp_name = f"DDQN_ads_p{args.num_particles}_g{args.num_subgroups}_{seed}"
    result_dir = os.path.join("results", exp_name)
    os.makedirs(result_dir, exist_ok=True)
    try:
        from envs.ad_scenario_env import ADScenarioEnv
        import gpytorch
    except ImportError:
        print("Error: ADS mode requires 'gpytorch'")
        exit(1)
    try:
        gp_model, gp_likelihood = load_surrogate_model(args.model_path)
    except Exception as e:
        print(f"❌ Error: {e}, using dummy model")
        gp_model, gp_likelihood = get_dummy_surrogate_model(args.dim)
    env = ADScenarioEnv(gp_model=gp_model, gp_likelihood=gp_likelihood,
                       dim=args.dim, bounds=tuple(args.bounds), runner=None)
    state_dim = get_state_aware_state_dim()
    action_dim = get_state_aware_action_dim()
    agent = DoubleDQNAgent(
        state_dim=state_dim, num_actions=action_dim, alpha=args.lr, gamma=args.gamma,
        buffer_size=args.buffer_size, batch_size=args.batch_size,
        epsilon=args.epsilon_start, min_epsilon=args.epsilon_end,
        epsilon_decay=args.epsilon_decay, temperature=args.temperature
    )
    pso = StateAwareNichePSO(
        env=env, agent=agent, num_particles=args.num_particles,
        num_subgroups=args.num_subgroups, max_iterations=args.iterations_per_episode,
        enable_restart=True, restart_patience=5, danger_threshold=args.danger_threshold,
        task_type='ads', action_interval=args.action_interval,
        action_smoothing_factor=args.action_smoothing,
        reward_calculator_class=SafetyRewardCalculator, use_subgroup_features=True
    )
    all_rewards, all_losses, all_q_values = [], [], []
    all_best_fitness, all_policy_entropies = [], []
    start_time = time.time()
    for episode in range(args.num_episodes):
        if episode == 0:
            pso.init_subgroups(reset_trackers=True)
        result = train_one_episode(pso, env, agent, args.iterations_per_episode)
        all_rewards.append(result['mean_reward'])
        all_losses.append(result['mean_loss'])
        all_q_values.append(result['mean_q_value'])
        all_best_fitness.append(result['best_fitness'])
        all_policy_entropies.append(result['avg_policy_entropy'])
        window = min(20, len(all_rewards))
        avg_reward = np.mean(all_rewards[-window:])
        elapsed = time.time() - start_time
        speed = (episode + 1) / elapsed
        remaining = (args.num_episodes - episode - 1) / speed if speed > 0 else 0
        danger_flag = "🚨" if result.get('danger_found', False) else "  "
        danger_count = result.get('num_danger_groups', 0)
        print(f"[Run {run_idx+1}] Ep {episode:3d}/{args.num_episodes} | "
              f"R: {result['mean_reward']:.3f} | Avg: {avg_reward:.3f} | "
              f"Fit: {result['best_fitness']:.4f} {danger_flag}({danger_count}) | "
              f"Loss: {result['mean_loss']:.4f} | ETA: {remaining/60:.1f}m")
        if episode > 0 and episode % 50 == 0:
             agent.save(os.path.join(result_dir, f"ddqn_model_ep{episode}.pth"))
    plot_path = os.path.join(result_dir, 'ddqn_training_curves.png')
    # DDQN plot 仅需简化版本的绘图
    plot_training_curves(all_rewards, all_losses, all_q_values, [], [], plot_path)
    agent.save(os.path.join(result_dir, f"ddqn_model_final.pth"))
    from scipy.io import savemat
    mat_path = os.path.join(result_dir, 'ddqn_training_data.mat')
    savemat(mat_path, {
        'algorithm': 'DDQN', 'episodes': np.arange(1, len(all_rewards) + 1),
        'mean_rewards': np.array(all_rewards), 'best_fitness': np.array(all_best_fitness),
        'losses': np.array(all_losses), 'q_values': np.array(all_q_values),
        'policy_dist_entropy': np.array(all_policy_entropies),
    })
    print(f"[Run {run_idx+1}] Training data saved to: {mat_path}")
    print(f"\n[Run {run_idx+1}] Reward Calculator Statistics:")
    pso.reward_calculator.print_stats()
    compute_rl_metrics(all_rewards, [], pso)
    run_time = time.time() - start_time
    print(f"[Run {run_idx+1}] Time: {run_time/60:.1f}m | Final Avg Reward: {np.mean(all_rewards[-20:]):.4f}")
    if all_policy_entropies:
        print(f"[Run {run_idx+1}] Final Avg Policy Entropy (last 20): {np.mean(all_policy_entropies[-20:]):.4f} nats")
    return {
        'seed': seed, 'rewards': np.array(all_rewards), 'losses': np.array(all_losses),
        'q_values': np.array(all_q_values), 'best_fitness': np.array(all_best_fitness),
        'policy_entropies': np.array(all_policy_entropies), 'run_time': run_time,
    }


def main():
    parser = argparse.ArgumentParser(description='DDQN RL-PSO Training (Multi-Run)')
    
    # 训练参数
    parser.add_argument('--num_episodes', type=int, default=500)
    parser.add_argument('--iterations_per_episode', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42, help='单次运行的种子 (仅 num_runs=1 时使用)')

    # PSO参数
    parser.add_argument('--num_particles', type=int, default=500)
    parser.add_argument('--num_subgroups', type=int, default=5)
    parser.add_argument('--dim', type=int, default=3)
    parser.add_argument('--bounds', type=float, nargs=2, default=[-1, 1])

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

    # 策略执行区间与平滑
    parser.add_argument('--action_interval', type=int, default=10,
                        help='Number of steps to hold the same action (Frame Skipping)')
    parser.add_argument('--action_smoothing', type=float, default=0.6,
                        help='Smoothing factor (alpha) for parameter updates')

    # ===== 多次独立实验参数 =====
    parser.add_argument('--num_runs', type=int, default=5,
                        help='独立实验次数 (默认5次, RL论文标准: 3~5次)')
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[42, 123, 456, 789, 1024],
                        help='每次实验的随机种子列表')

    args = parser.parse_args()

    # 确保种子列表长度 >= num_runs
    if len(args.seeds) < args.num_runs:
        extra_seeds = [args.seeds[-1] + i * 111 for i in range(1, args.num_runs - len(args.seeds) + 1)]
        args.seeds = args.seeds + extra_seeds
    args.seeds = args.seeds[:args.num_runs]
    
    print("=" * 60)
    print("DDQN RL-PSO Training (Multi-Run)")
    print("=" * 60)
    print(f"  独立实验次数 (num_runs): {args.num_runs}")
    print(f"  随机种子列表: {args.seeds}")
    print(f"  每次实验 Episodes: {args.num_episodes}")
    print("=" * 60)

    all_run_results = []
    total_start = time.time()

    for run_idx in range(args.num_runs):
        seed = args.seeds[run_idx]
        result = run_single_experiment(args, run_idx, seed)
        all_run_results.append(result)
        print(f"\n✅ Run {run_idx + 1}/{args.num_runs} 完成 (seed={seed})")

    total_time = time.time() - total_start

    script_dir = os.path.dirname(os.path.abspath(__file__))
    matlab_dir = os.path.normpath(os.path.join(script_dir, '..', '..', '..', 'matlab_scripts'))
    os.makedirs(matlab_dir, exist_ok=True)

    num_eps = args.num_episodes
    num_runs = args.num_runs

    mat_rewards = np.zeros((num_runs, num_eps))
    mat_best_fitness = np.zeros((num_runs, num_eps))
    mat_losses = np.zeros((num_runs, num_eps))
    mat_q_values = np.zeros((num_runs, num_eps))
    mat_policy_entropies = np.zeros((num_runs, num_eps))

    for i, res in enumerate(all_run_results):
        n = min(len(res['rewards']), num_eps)
        mat_rewards[i, :n] = res['rewards'][:n]
        mat_best_fitness[i, :n] = res['best_fitness'][:n]
        mat_losses[i, :n] = res['losses'][:n]
        mat_q_values[i, :n] = res['q_values'][:n]
        mat_policy_entropies[i, :n] = res['policy_entropies'][:n]

    from scipy.io import savemat
    multi_run_mat_path = os.path.join(matlab_dir, 'ddqn_multi_run_data.mat')
    savemat(multi_run_mat_path, {
        'algorithm': 'DDQN', 'num_runs': num_runs, 'num_episodes': num_eps,
        'seeds': np.array(args.seeds), 'episodes': np.arange(1, num_eps + 1),
        'all_rewards': mat_rewards, 'all_best_fitness': mat_best_fitness,
        'all_losses': mat_losses, 'all_q_values': mat_q_values,
        'all_policy_entropies': mat_policy_entropies,
        'mean_rewards': np.mean(mat_rewards, axis=0), 'std_rewards': np.std(mat_rewards, axis=0),
        'mean_best_fitness': np.mean(mat_best_fitness, axis=0), 'std_best_fitness': np.std(mat_best_fitness, axis=0),
    })
    print(f"\n📊 汇总数据已保存到: {multi_run_mat_path}")

    for i, res in enumerate(all_run_results):
        run_mat_path = os.path.join(matlab_dir, f'ddqn_run_{i+1}.mat')
        savemat(run_mat_path, {
            'algorithm': 'DDQN', 'seed': res['seed'],
            'episodes': np.arange(1, len(res['rewards']) + 1),
            'mean_rewards': res['rewards'], 'best_fitness': res['best_fitness'],
            'losses': res['losses'], 'q_values': res['q_values'],
            'policy_dist_entropy': res['policy_entropies'],
        })
    print(f"📊 各 run 独立数据已保存: ddqn_run_1.mat ~ ddqn_run_{num_runs}.mat")

    print("\n" + "=" * 60)
    print("🏁 Multi-Run Training Complete!")
    print("=" * 60)
    print(f"  总运行次数: {num_runs}")
    print(f"  总耗时: {total_time/60:.1f} min")

    final_rewards = [np.mean(res['rewards'][-20:]) for res in all_run_results]
    print(f"  最终平均奖励 (last 20 eps):")
    for i, (seed, fr) in enumerate(zip(args.seeds, final_rewards)):
        print(f"    Run {i+1} (seed={seed}): {fr:.4f}")
    print(f"  跨 Run 均值 ± 标准差: {np.mean(final_rewards):.4f} ± {np.std(final_rewards):.4f}")
    print(f"\n  汇总文件: {multi_run_mat_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
