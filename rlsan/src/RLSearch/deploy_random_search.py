"""
deploy_random_search.py - Random策略危险工况搜索

用于对比实验：
- 使用与RL版本（deploy_ad_search.py）相同的StateAwareNichePSO框架
- 将RL Agent替换为RandomAgent（均匀随机选择动作）
- 保持所有其他参数一致

对比策略：
1. RL策略: deploy_ad_search.py（使用训练好的权重）
2. 固定参数PSO: pure_niche_pso.py（固定w, c1, c2）
3. Random策略: 本脚本（随机选择动作）
"""

import argparse
import numpy as np
import torch
import os
import time
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 本地导入
from rl_core.random_agent import RandomAgent
from envs.ad_scenario_env import ADScenarioEnv
from optimizer.niche_pso import StateAwareNichePSO
from utils import set_seed, load_surrogate_model




def select_representative_seeds(hazardous_points: np.ndarray, 
                                 fitness_values: np.ndarray,
                                 niche_radius: float = 0.05,
                                 niche_capacity: int = 3,
                                 top_k_dangerous: int = 20) -> dict:
    """
    筛选最危险和最有代表性的危险种子
    """
    if len(hazardous_points) == 0:
        return {'top_dangerous': np.array([]), 
                'representative': np.array([]),
                'top_dangerous_fitness': np.array([]),
                'representative_fitness': np.array([])}
    
    # 去重
    _, unique_idx = np.unique(np.round(hazardous_points, decimals=4), axis=0, return_index=True)
    unique_points = hazardous_points[unique_idx]
    unique_fitness = fitness_values[unique_idx]
    
    print(f"  Unique points: {len(unique_points)}")
    
    # Top-K 最危险点
    sorted_idx = np.argsort(unique_fitness)
    top_k = min(top_k_dangerous, len(unique_points))
    top_dangerous_idx = sorted_idx[:top_k]
    top_dangerous = unique_points[top_dangerous_idx]
    top_dangerous_fitness = unique_fitness[top_dangerous_idx]
    
    print(f"  Top {top_k} dangerous selected")
    
    # 基于小生境覆盖的代表性种子
    n_unique = len(unique_points)
    coverage_counts = np.zeros(n_unique)
    used = np.zeros(n_unique, dtype=bool)
    representative = []
    representative_fitness = []
    
    for i in sorted_idx:
        if used[i] or coverage_counts[i] >= niche_capacity:
            continue
        
        representative.append(unique_points[i])
        representative_fitness.append(unique_fitness[i])
        used[i] = True
        
        distances = np.linalg.norm(unique_points - unique_points[i], axis=1)
        coverage_counts[distances < niche_radius] += 1
    
    representative = np.array(representative) if representative else np.array([])
    representative_fitness = np.array(representative_fitness) if representative_fitness else np.array([])
    
    print(f"  Representative seeds: {len(representative)}")
    
    return {
        'top_dangerous': top_dangerous,
        'top_dangerous_fitness': top_dangerous_fitness,
        'representative': representative,
        'representative_fitness': representative_fitness,
    }


def plot_seeds(seeds: np.ndarray, fitness: np.ndarray, title: str, save_path: str):
    """种子可视化"""
    if len(seeds) == 0:
        print(f"  No seeds to plot for {title}")
        return
    
    dim = seeds.shape[1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. 2D 散点图
    if dim >= 2:
        scatter = axes[0].scatter(seeds[:, 0], seeds[:, 1], 
                                   c=fitness, cmap='RdYlGn_r', 
                                   s=60, edgecolors='black', linewidths=0.5)
        plt.colorbar(scatter, ax=axes[0], label='Fitness')
        axes[0].set_xlabel('Dim 1')
        axes[0].set_ylabel('Dim 2')
        axes[0].set_title(f'{title} (N={len(seeds)})')
        axes[0].grid(True, linestyle='--', alpha=0.3)
    
    # 2. 适应度分布
    axes[1].hist(fitness, bins=min(20, len(fitness)), color='coral', 
                 alpha=0.7, edgecolor='black')
    axes[1].axvline(np.min(fitness), color='red', linestyle='--', 
                    label=f'Min: {np.min(fitness):.4f}')
    axes[1].set_xlabel('Fitness')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Fitness Distribution')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.3)
    
    # 3. 各维度分布
    for d in range(min(dim, 5)):
        axes[2].hist(seeds[:, d], bins=20, alpha=0.5, label=f'Dim {d+1}')
    axes[2].set_xlabel('Value')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Parameter Distribution')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Random Strategy AD Scenario Search')
    
    # 模型路径（Random策略不使用预训练权重）
    parser.add_argument('--model_path', type=str, default='../../results/s1exp/surrogate_model.pkl')
    
    # 搜索配置
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--dim', type=int, default=3)
    parser.add_argument('--bounds', type=float, nargs=2, default=[-1, 1])
    
    # PSO配置（与RL版本保持一致）
    parser.add_argument('--num_particles', type=int, default=2000)
    parser.add_argument('--num_subgroups', type=int, default=20)
    parser.add_argument('--init_niche_radius', type=float, default=0.2)
    
    # 筛选阈值
    parser.add_argument('--danger_threshold', type=float, default=-0.3)
    parser.add_argument('--uncertainty_threshold', type=float, default=0.05)
    
    # 随机种子和输出
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./random_search_results')
    parser.add_argument('--use_carla', action='store_true')
    
    # 种子筛选参数
    parser.add_argument('--niche_radius', type=float, default=0.2)
    parser.add_argument('--niche_capacity', type=int, default=3)
    parser.add_argument('--top_k', type=int, default=20)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Random Strategy AD Scenario Search")
    print("=" * 60)
    print("Note: This uses RANDOM action selection for comparison with RL")
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载环境
    print("\n[1] Loading environment...")
    try:
        gp_model, gp_likelihood = load_surrogate_model(args.model_path)
        print("  Surrogate model loaded")
    except Exception as e:
        print(f"  Error: {e}")
        gp_model, gp_likelihood = None, None
    
    if gp_model is None:
        print("  Error: Surrogate model is required. Cannot proceed without it.")
        exit(1)
    
    env = ADScenarioEnv(
        gp_model=gp_model, gp_likelihood=gp_likelihood,
        runner=None, dim=args.dim, bounds=tuple(args.bounds),
        uncertainty_threshold=args.uncertainty_threshold
    )
    
    # 创建 Random Agent
    print("\n[2] Creating Random Agent...")
    agent = RandomAgent(
        state_dim=25,  # 与StateAwareNichePSO.STATE_DIM一致
        num_actions=6,  # 与StateAwareActionSpace一致
        seed=args.seed
    )
    print(f"  {agent}")
    
    # 运行搜索
    print("\n[3] Running search with Random strategy...")
    pso = StateAwareNichePSO(
        env=env, agent=agent,
        num_particles=args.num_particles,
        num_subgroups=args.num_subgroups,
        max_iterations=args.iterations,
        enable_restart=True,
        restart_patience=5
    )
    pso.init_subgroups(reset_trackers=True)
    
    start_time = time.time()
    
    # 运行搜索
    hazardous_points = []
    action_history = []  # 记录动作分布
    
    for iteration in range(args.iterations):
        pso.run_one_iteration()
        
        # 收集低于阈值的危险点
        for g in pso.subgroups:
            if g['global_best_fitness'] < args.danger_threshold:
                hazardous_points.append(g['global_best_position'].copy())
        
        # 记录当前动作
        action_history.extend(pso.current_actions)
        
        if (iteration + 1) % 100 == 0:
            best_fitness = min(g['global_best_fitness'] for g in pso.subgroups)
            print(f"  Iter {iteration+1}/{args.iterations} | Best: {best_fitness:.4f} | Hazardous: {len(hazardous_points)}")
    
    hazardous_points = np.array(hazardous_points) if hazardous_points else np.array([])
    search_time = time.time() - start_time
    
    print(f"\n  Search time: {search_time:.2f}s")
    print(f"  Raw hazardous points: {len(hazardous_points)}")
    
    # 打印动作分布统计
    print(f"\n  Agent Stats: {agent.get_stats()}")
    
    if len(hazardous_points) == 0:
        print("\nNo hazardous points found. Consider lowering danger_threshold.")
        return
    
    # 筛选种子
    print("\n[4] Filtering seeds...")
    fitness_values, _ = env.evaluate(hazardous_points)
    
    seeds_dict = select_representative_seeds(
        hazardous_points, fitness_values,
        niche_radius=args.niche_radius,
        niche_capacity=args.niche_capacity,
        top_k_dangerous=args.top_k
    )
    
    # 可视化
    print("\n[5] Generating visualization...")
    plot_seeds(seeds_dict['representative'], seeds_dict['representative_fitness'],
               'Random - Representative Seeds', os.path.join(args.output_dir, 'random_representative_seeds.png'))
    plot_seeds(seeds_dict['top_dangerous'], seeds_dict['top_dangerous_fitness'],
               'Random - Top Dangerous Seeds', os.path.join(args.output_dir, 'random_top_dangerous_seeds.png'))
    
    # 保存结果
    print("\n[6] Saving results...")
    np.save(os.path.join(args.output_dir, 'random_representative_seeds.npy'), 
            seeds_dict['representative'])
    np.save(os.path.join(args.output_dir, 'random_representative_fitness.npy'), 
            seeds_dict['representative_fitness'])
    np.save(os.path.join(args.output_dir, 'random_top_dangerous_seeds.npy'), 
            seeds_dict['top_dangerous'])
    np.save(os.path.join(args.output_dir, 'random_top_dangerous_fitness.npy'), 
            seeds_dict['top_dangerous_fitness'])
    
    with open(os.path.join(args.output_dir, 'random_search_results.pkl'), 'wb') as f:
        pickle.dump({
            'seeds_dict': seeds_dict,
            'search_time': search_time,
            'args': vars(args),
            'agent_stats': agent.get_stats(),
            'action_history': action_history,
        }, f)
    
    print(f"  Results saved to: {args.output_dir}")
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("SUMMARY (Random Strategy)")
    print("=" * 60)
    print(f"Representative seeds: {len(seeds_dict['representative'])}")
    print(f"Top dangerous seeds:  {len(seeds_dict['top_dangerous'])}")
    
    if len(seeds_dict['top_dangerous']) > 0:
        print(f"\nMost dangerous scenario:")
        print(f"  Fitness:  {seeds_dict['top_dangerous_fitness'][0]:.6f}")
        print(f"  Position: {seeds_dict['top_dangerous'][0]}")
    
    # 对比提示
    print("\n" + "-" * 60)
    print("To compare with other strategies:")
    print("  RL:        python deploy_ad_search.py --output_dir ./rl_search_results")
    print("  Fixed PSO: python pure_niche_pso.py")
    print("  Random:    python deploy_random_search.py --output_dir ./random_search_results")
    print("=" * 60)


if __name__ == '__main__':
    main()
