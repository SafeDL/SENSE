"""
train_ppo.py - PPO专用训练脚本（多运行版）

支持多次独立实验、输出平均指标±标准差：
1. 使用SafetyRewardCalculator统一奖励计算
2. 实现熵系数自适应调度
3. 多运行模式支持统计结果

使用方法：
    # 单次运行
    python train_ppo.py --num_episodes 200 --num_runs 1

    # 多次独立实验 (RL标准做法: 3~5次)
    python train_ppo.py --num_episodes 200 --num_runs 5 --seeds 42 123 456 789 1024

    # 汇总数据输出到 matlab_scripts/ 用于对比分析
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
from rl_core.ppo_agent import PPOAgent
from optimizer.niche_pso import StateAwareNichePSO, get_state_aware_state_dim
from optimizer.reward import SafetyRewardCalculator
from utils import set_seed, load_surrogate_model, get_dummy_surrogate_model, compute_rl_metrics

class EntropyScheduler:
    """
    熵系数调度器
    
    实现熵系数随训练进度自适应衰减：
    - 早期：高熵鼓励探索
    - 后期：低熵促进收敛
    """
    
    def __init__(self, 
                 initial_coef: float = 0.02,
                 final_coef: float = 0.001,
                 warmup_episodes: int = 10,
                 decay_type: str = 'linear'):
        """
        Args:
            initial_coef: 初始熵系数
            final_coef: 最终熵系数
            warmup_episodes: 预热期（保持初始值）
            decay_type: 衰减类型 ('linear', 'cosine', 'exp')
        """
        self.initial_coef = initial_coef
        self.final_coef = final_coef
        self.warmup_episodes = warmup_episodes
        self.decay_type = decay_type
    
    def get_coef(self, episode: int, total_episodes: int) -> float:
        """
        获取当前熵系数
        
        Args:
            episode: 当前episode
            total_episodes: 总episode数
            
        Returns:
            current_coef: 当前熵系数
        """
        if episode < self.warmup_episodes:
            return self.initial_coef
        
        # 计算衰减进度
        decay_progress = (episode - self.warmup_episodes) / max(1, total_episodes - self.warmup_episodes)
        decay_progress = min(1.0, decay_progress)
        
        if self.decay_type == 'linear':
            coef = self.initial_coef + (self.final_coef - self.initial_coef) * decay_progress
        elif self.decay_type == 'cosine':
            coef = self.final_coef + 0.5 * (self.initial_coef - self.final_coef) * (1 + np.cos(np.pi * decay_progress))
        elif self.decay_type == 'exp':
            decay_rate = np.log(self.final_coef / self.initial_coef)
            coef = self.initial_coef * np.exp(decay_rate * decay_progress)
        else:
            coef = self.initial_coef
        
        return max(self.final_coef, coef)


def train_one_episode(pso: StateAwareNichePSO,
                      env,
                      agent: PPOAgent,
                      iterations_per_episode: int = 100) -> dict:
    """训练单个episode（连续动作）"""
    
    episode_rewards = []
    episode_actions = []
    episode_entropies = []  # 每步策略熵
    
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
    
    # 刷新所有并行的轨迹缓存，并标记为Done
    if hasattr(pso, 'flush_experience_to_agent'):
        pso.flush_experience_to_agent(done_all=True)
    
    # Episode结束时PPO更新
    update_info = agent.update(next_value=0.0) if hasattr(agent, 'update') else {
        'policy_loss': 0, 'value_loss': 0, 'entropy': 0
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
    
    # 计算连续动作参数的均値
    if episode_actions:
        action_means = np.mean(episode_actions, axis=0)
    else:
        action_means = np.zeros(4)
    
    # 计算最后20步的平均策略分布熵
    window = 20
    if episode_entropies:
        recent_entropies = episode_entropies[-window:]
        avg_policy_entropy = float(np.mean(recent_entropies))
    else:
        avg_policy_entropy = 0.0
    
    return {
        'mean_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
        'policy_loss': update_info.get('policy_loss', 0.0),
        'value_loss': update_info.get('value_loss', 0.0),
        'entropy': update_info.get('entropy', 0.0),
        'best_fitness': best_fitness,
        'num_niches': stats['num_niches'],
        'coverage_rate': stats['coverage_rate'],
        'action_means': action_means,
        'episode_actions': episode_actions,
        'milestones_achieved': reward_stats.get('milestones_achieved', 0),
        'best_fitness_ever': reward_stats.get('best_fitness', float('inf')),
        'danger_found': danger_found,
        'num_danger_groups': num_danger_groups,
        'avg_policy_entropy': avg_policy_entropy,   # 新增：最后20步平均策略熵
    }


def plot_training_curves(rewards: list, policy_losses: list, value_losses: list,
                         entropies: list, action_history: list, entropy_coefs: list,
                         save_path: str):
    """绘制PPO V2训练曲线（增强版）"""
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
    
    # 3. Entropy + Entropy Coef
    ax3 = axes[0, 2]
    valid = [e for e in entropies if e is not None and not np.isnan(e)]
    if valid:
        ax3.plot(valid, color='purple', alpha=0.5, label='Entropy')
        if len(valid) >= window:
            smoothed = np.convolve(valid, np.ones(window)/window, mode='valid')
            ax3.plot(range(window-1, len(valid)), smoothed, color='purple', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Entropy', color='purple')
    ax3.tick_params(axis='y', labelcolor='purple')
    
    # 熵系数（右轴）
    ax3_twin = ax3.twinx()
    if entropy_coefs:
        ax3_twin.plot(entropy_coefs, color='orange', linestyle='--', label='Entropy Coef')
    ax3_twin.set_ylabel('Entropy Coef', color='orange')
    ax3_twin.tick_params(axis='y', labelcolor='orange')
    ax3.set_title('Policy Entropy + Coefficient Schedule')
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


def run_single_experiment(args, run_idx: int, seed: int) -> dict:
    """
    运行一次完整的 PPO 训练实验。

    Args:
        args: 命令行参数
        run_idx: 当前 run 的索引 (0-based)
        seed: 本次实验的随机种子

    Returns:
        dict: 包含本次实验全部训练历史的字典
    """
    print(f"\n{'#' * 60}")
    print(f"# Run {run_idx + 1} | Seed: {seed}")
    print(f"{'#' * 60}")

    set_seed(seed)

    # 路径设置（每个 run 独立保存模型）
    exp_name = f"PPO_ads_p{args.num_particles}_g{args.num_subgroups}_{seed}"
    result_dir = os.path.join("results", exp_name)
    os.makedirs(result_dir, exist_ok=True)

    # 创建环境 (仅支持 ADS 自动驾驶场景)
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

    # 创建熵调度器
    entropy_scheduler = EntropyScheduler(
        initial_coef=args.initial_entropy_coef,
        final_coef=args.final_entropy_coef,
        warmup_episodes=args.entropy_warmup,
        decay_type=args.entropy_decay_type
    )

    # 创建连续动作PPO Agent（每次 run 全新初始化）
    agent = PPOAgent(
        state_dim=get_state_aware_state_dim(),
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        entropy_coef=args.initial_entropy_coef,
        hidden_dims=[128, 128, 64]
    )

    # 创建PPO专用PSO（每次 run 全新初始化）
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

    all_rewards, all_policy_losses, all_value_losses, all_entropies = [], [], [], []
    all_action_means = []
    all_entropy_coefs = []
    all_best_fitness = []
    all_policy_entropies = []

    start_time = time.time()

    for episode in range(args.num_episodes):
        func_name = env.get_current_function_name()

        # 更新熵系数
        current_entropy_coef = entropy_scheduler.get_coef(episode, args.num_episodes)
        agent.entropy_coef = current_entropy_coef
        all_entropy_coefs.append(current_entropy_coef)

        if episode == 0:
            pso.init_subgroups(reset_trackers=True)

        result = train_one_episode(pso, env, agent, args.iterations_per_episode)

        all_rewards.append(result['mean_reward'])
        all_policy_losses.append(result['policy_loss'])
        all_value_losses.append(result['value_loss'])
        all_entropies.append(result['entropy'])
        all_action_means.append(result['action_means'])
        all_best_fitness.append(result['best_fitness'])
        all_policy_entropies.append(result['avg_policy_entropy'])

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

        print(f"[Run {run_idx+1}] Ep {episode:3d}/{args.num_episodes} | {func_name:<15} | "
              f"R: {result['mean_reward']:.3f} | Avg: {avg_reward:.3f} | "
              f"Fit: {best_fit:.4f} {danger_flag}({danger_count}) | "
              f"Ent: {result['entropy']:.3f} | DistEnt: {result['avg_policy_entropy']:.3f} | "
              f"{action_str} | ETA: {remaining/60:.1f}m")

        if episode > 0 and episode % 50 == 0:
             agent.save(os.path.join(result_dir, f"ppo_model_ep{episode}.pth"))

    # 绘制本次 run 的训练曲线
    plot_path = os.path.join(result_dir, 'ppo_training_curves.png')
    plot_training_curves(all_rewards, all_policy_losses, all_value_losses,
                         all_entropies, all_action_means, all_entropy_coefs, plot_path)

    agent.save(os.path.join(result_dir, "ppo_model_final.pth"))

    # 导出本次 run 训练数据为 .mat 格式
    action_array = np.array(all_action_means)
    mat_path = os.path.join(result_dir, 'ppo_training_data.mat')
    savemat(mat_path, {
        'algorithm': 'PPO',
        'episodes': np.arange(1, len(all_rewards) + 1),
        'mean_rewards': np.array(all_rewards),
        'best_fitness': np.array(all_best_fitness),
        'policy_losses': np.array(all_policy_losses),
        'value_losses': np.array(all_value_losses),
        'entropies': np.array(all_entropies),
        'entropy_coefs': np.array(all_entropy_coefs),
        'policy_dist_entropy': np.array(all_policy_entropies),
        'action_w': action_array[:, 0] if len(action_array) > 0 else np.array([]),
        'action_c1': action_array[:, 1] if len(action_array) > 0 else np.array([]),
        'action_c2': action_array[:, 2] if len(action_array) > 0 else np.array([]),
        'action_vs': action_array[:, 3] if len(action_array) > 0 else np.array([]),
    })
    print(f"[Run {run_idx+1}] Training data saved to: {mat_path}")

    # 打印奖励计算器统计
    print(f"\n[Run {run_idx+1}] Reward Calculator Statistics:")
    pso.reward_calculator.print_stats()

    # 评测并输出比较指标
    compute_rl_metrics(all_rewards, all_action_means, pso)

    run_time = time.time() - start_time
    print(f"[Run {run_idx+1}] Time: {run_time/60:.1f}m | "
          f"Final Avg Reward: {np.mean(all_rewards[-20:]):.4f}")
    if all_policy_entropies:
        print(f"[Run {run_idx+1}] Final Avg Policy Entropy (last 20): "
              f"{np.mean(all_policy_entropies[-20:]):.4f} nats")

    return {
        'seed': seed,
        'rewards': np.array(all_rewards),
        'policy_losses': np.array(all_policy_losses),
        'value_losses': np.array(all_value_losses),
        'entropies': np.array(all_entropies),
        'best_fitness': np.array(all_best_fitness),
        'policy_entropies': np.array(all_policy_entropies),
        'action_w': action_array[:, 0] if len(action_array) > 0 else np.array([]),
        'action_c1': action_array[:, 1] if len(action_array) > 0 else np.array([]),
        'action_c2': action_array[:, 2] if len(action_array) > 0 else np.array([]),
        'action_vs': action_array[:, 3] if len(action_array) > 0 else np.array([]),
        'run_time': run_time,
    }


def main():
    parser = argparse.ArgumentParser(description='PPO V2 RL-PSO Training (Multi-Run)')

    # 训练参数
    parser.add_argument('--num_episodes', type=int, default=500)
    parser.add_argument('--iterations_per_episode', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42, help='单次运行的种子 (仅 num_runs=1 时使用)')

    # PSO参数
    parser.add_argument('--num_particles', type=int, default=500)
    parser.add_argument('--num_subgroups', type=int, default=10)
    parser.add_argument('--dim', type=int, default=3)
    parser.add_argument('--bounds', type=float, nargs=2, default=[-1, 1])

    # PPO参数（优化用于危险场景搜索）
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.7)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--clip_epsilon', type=float, default=0.2)
    parser.add_argument('--ppo_epochs', type=int, default=15)
    parser.add_argument('--mini_batch_size', type=int, default=128)

    # 熵调度参数
    parser.add_argument('--initial_entropy_coef', type=float, default=0.02)
    parser.add_argument('--final_entropy_coef', type=float, default=0.001)
    parser.add_argument('--entropy_warmup', type=int, default=10)
    parser.add_argument('--entropy_decay_type', type=str, default='linear',
                        choices=['linear', 'cosine', 'exp'])

    parser.add_argument('--danger_threshold', type=float, default=-0.3, help='Threshold for niche discovery')

    # ADS params
    parser.add_argument('--model_path', type=str, default='../../results/s1exp/surrogate_model.pkl')

    # 策略执行区间
    parser.add_argument('--action_interval', type=int, default=10)
    parser.add_argument('--action_smoothing', type=float, default=0.6)

    # ===== 多次独立实验参数 =====
    parser.add_argument('--num_runs', type=int, default=5,
                        help='独立实验次数 (默认5次, RL论文标准: 3~5次)')
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[42, 123, 456, 789, 1024],
                        help='每次实验的随机种子列表')

    args = parser.parse_args()

    # 确保种子列表长度 >= num_runs
    if len(args.seeds) < args.num_runs:
        # 自动扩展种子列表
        extra_seeds = [args.seeds[-1] + i * 111 for i in range(1, args.num_runs - len(args.seeds) + 1)]
        args.seeds = args.seeds + extra_seeds
    args.seeds = args.seeds[:args.num_runs]
    
    print("=" * 60)
    print("PPO V2 RL-PSO Training (Multi-Run)")
    print("=" * 60)
    print(f"  独立实验次数 (num_runs): {args.num_runs}")
    print(f"  随机种子列表: {args.seeds}")
    print(f"  每次实验 Episodes: {args.num_episodes}")
    print(f"  action_interval: {args.action_interval}")
    print(f"  action_smoothing: {args.action_smoothing}")
    print("=" * 60)

    # ===== 多次独立实验循环 =====
    all_run_results = []
    total_start = time.time()

    for run_idx in range(args.num_runs):
        seed = args.seeds[run_idx]
        result = run_single_experiment(args, run_idx, seed)
        all_run_results.append(result)
        print(f"\n✅ Run {run_idx + 1}/{args.num_runs} 完成 (seed={seed})")

    total_time = time.time() - total_start

    # ===== 汇总所有 runs 的数据并保存到 matlab_scripts/ =====
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # matlab_scripts 位于 SENSE 根目录下
    matlab_dir = os.path.normpath(os.path.join(script_dir, '..', '..', '..', 'matlab_scripts'))
    os.makedirs(matlab_dir, exist_ok=True)

    num_eps = args.num_episodes
    num_runs = args.num_runs

    # 构建矩阵：每行一个 run，每列一个 episode (runs × episodes)
    mat_rewards = np.zeros((num_runs, num_eps))
    mat_action_w = np.zeros((num_runs, num_eps))
    mat_action_c1 = np.zeros((num_runs, num_eps))
    mat_action_c2 = np.zeros((num_runs, num_eps))
    mat_action_vs = np.zeros((num_runs, num_eps))
    mat_best_fitness = np.zeros((num_runs, num_eps))
    mat_policy_losses = np.zeros((num_runs, num_eps))
    mat_value_losses = np.zeros((num_runs, num_eps))
    mat_policy_entropies = np.zeros((num_runs, num_eps))

    for i, res in enumerate(all_run_results):
        n = min(len(res['rewards']), num_eps)
        mat_rewards[i, :n] = res['rewards'][:n]
        mat_action_w[i, :n] = res['action_w'][:n]
        mat_action_c1[i, :n] = res['action_c1'][:n]
        mat_action_c2[i, :n] = res['action_c2'][:n]
        mat_action_vs[i, :n] = res['action_vs'][:n]
        mat_best_fitness[i, :n] = res['best_fitness'][:n]
        mat_policy_losses[i, :n] = res['policy_losses'][:n]
        mat_value_losses[i, :n] = res['value_losses'][:n]
        mat_policy_entropies[i, :n] = res['policy_entropies'][:n]

    # 保存汇总 .mat 文件
    multi_run_mat_path = os.path.join(matlab_dir, 'ppo_multi_run_data.mat')
    savemat(multi_run_mat_path, {
        'algorithm': 'PPO',
        'num_runs': num_runs,
        'num_episodes': num_eps,
        'seeds': np.array(args.seeds),
        'episodes': np.arange(1, num_eps + 1),
        # 矩阵数据 (runs × episodes)
        'all_rewards': mat_rewards,
        'all_action_w': mat_action_w,
        'all_action_c1': mat_action_c1,
        'all_action_c2': mat_action_c2,
        'all_action_vs': mat_action_vs,
        'all_best_fitness': mat_best_fitness,
        'all_policy_losses': mat_policy_losses,
        'all_value_losses': mat_value_losses,
        'all_policy_entropies': mat_policy_entropies,
        # 统计量 (方便 MATLAB 直接使用)
        'mean_rewards': np.mean(mat_rewards, axis=0),
        'std_rewards': np.std(mat_rewards, axis=0),
        'mean_action_w': np.mean(mat_action_w, axis=0),
        'std_action_w': np.std(mat_action_w, axis=0),
        'mean_action_c1': np.mean(mat_action_c1, axis=0),
        'std_action_c1': np.std(mat_action_c1, axis=0),
        'mean_action_c2': np.mean(mat_action_c2, axis=0),
        'std_action_c2': np.std(mat_action_c2, axis=0),
        'mean_action_vs': np.mean(mat_action_vs, axis=0),
        'std_action_vs': np.std(mat_action_vs, axis=0),
    })
    print(f"\n📊 汇总数据已保存到: {multi_run_mat_path}")

    # 同时保存每个 run 的单独 .mat（与原有格式兼容）
    for i, res in enumerate(all_run_results):
        run_mat_path = os.path.join(matlab_dir, f'ppo_run_{i+1}.mat')
        savemat(run_mat_path, {
            'algorithm': 'PPO',
            'seed': res['seed'],
            'episodes': np.arange(1, len(res['rewards']) + 1),
            'mean_rewards': res['rewards'],
            'best_fitness': res['best_fitness'],
            'policy_losses': res['policy_losses'],
            'value_losses': res['value_losses'],
            'policy_dist_entropy': res['policy_entropies'],
            'action_w': res['action_w'],
            'action_c1': res['action_c1'],
            'action_c2': res['action_c2'],
            'action_vs': res['action_vs'],
        })
    print(f"📊 各 run 独立数据已保存: ppo_run_1.mat ~ ppo_run_{num_runs}.mat")

    # ===== 打印汇总 =====
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
    print(f"  请在 MATLAB 中运行 plot_ppo_multi_run.m 绘制可视化图")
    print("=" * 60)


if __name__ == '__main__':
    main()