"""
deploy_ads_search.py - Stage 2: 在线迁移自动驾驶危险工况搜索
"""

import argparse
import numpy as np
import torch
import os
import time
import pickle
import matplotlib
import gpytorch
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 本地导入
from rl_core.dqn_agent import DoubleDQNAgent
from rl_core.ppo_agent import PPOAgent
from rl_core.sac_agent import SACAgent
from rl_core.ddpg_agent import DDPGAgent
from rl_core.td3_agent import TD3Agent
from rl_core.trpo_agent import TRPOAgent
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
    """简化的种子可视化"""
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
    print(f" Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='AD Scenario Search')
    
    parser.add_argument('--pretrained', type=str, default='../../results/s1exp/sac_model_final.pth')
    parser.add_argument('--agent_type', type=str, default='sac', choices=['ddqn', 'ppo', 'sac', 'ddpg', 'td3', 'trpo'], help='Type of RL agent to load')
    parser.add_argument('--model_path', type=str, default='../../results/s1exp/surrogate_model.pkl')
    
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--dim', type=int, default=3)
    parser.add_argument('--bounds', type=float, nargs=2, default=[-1, 1])
    
    parser.add_argument('--num_particles', type=int, default=2000)
    parser.add_argument('--num_subgroups', type=int, default=20)
    parser.add_argument('--init_niche_radius', type=float, default=0.2)
    
    parser.add_argument('--danger_threshold', type=float, default=-0.3)
    parser.add_argument('--uncertainty_threshold', type=float, default=0.05)
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='./ad_search_results')
    parser.add_argument('--use_carla', action='store_true')
    
    parser.add_argument('--niche_radius', type=float, default=0.05)
    parser.add_argument('--niche_capacity', type=int, default=3)
    parser.add_argument('--top_k', type=int, default=20)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AD Scenario Search with Pretrained RL Agent")
    print("=" * 60)
    
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    import scipy.io as sio

    # 加载环境
    print("\n[1] Loading environment...")
    
    # 两种验证方式支持：
    # 1. 完全依赖代理模型 (use_carla=False): runner = None
    # 2. 结合真实仿真 (use_carla=True): runner = CarlaRunner()
    runner = None
    if args.use_carla:
        print("  Initializing CARLA runner for active simulation...")
        try:
            from envs.carla_runner import CarlaRunner
            runner = CarlaRunner()
            print("  [✓] CARLA runner successfully loaded.")
        except ImportError:
            print("  [!] Warning: CarlaRunner not found. Falling back to surrogate-only mode.")
            runner = None
    else:
        print("  [i] Surrogate-only mode (No real simulation).")
            
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
        runner=runner, dim=args.dim, bounds=tuple(args.bounds),
        uncertainty_threshold=args.uncertainty_threshold
    )
    
    # 加载 Agent
    print(f"\n[2] Loading agent (Type: {args.agent_type})...")
    try:
        if args.agent_type == 'ddqn':
            agent = DoubleDQNAgent.load(args.pretrained, freeze=True)
        elif args.agent_type == 'ppo':
            agent = PPOAgent.load(args.pretrained, freeze=True)
        elif args.agent_type == 'sac':
            agent = SACAgent.load(args.pretrained, freeze=True)
        elif args.agent_type == 'ddpg':
            agent = DDPGAgent.load(args.pretrained, freeze=True)
        elif args.agent_type == 'td3':
            agent = TD3Agent.load(args.pretrained, freeze=True)
        elif args.agent_type == 'trpo':
            agent = TRPOAgent.load(args.pretrained, freeze=True)
        print("  Agent loaded successfully")
    except Exception as e:
        print(f"  Error: {e}, using random policy (No Agent)")
        agent = None
    
    # 运行搜索
    print("\n[3] Running search...")
    # 使用 StateAwareNichePSO 以匹配训练时的配置
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
    
    # Metrics tracking
    budget_history = []
    risk_score_history = []  # Max Risk Score
    avg_risk_score_history = [] # Avg Risk Score
    ncs_history = []
    time_to_first_failure = -1
    
    # 运行搜索
    hazardous_points = []
    for iteration in range(args.iterations):
        pso.run_one_iteration()
        
        # 收集低于阈值的危险点
        current_hazardous = []
        best_fitness_iter = float('inf')
        total_fitness_iter = 0.0
        
        for g in pso.subgroups:
            fitness = g['global_best_fitness']
            total_fitness_iter += fitness
            if fitness < best_fitness_iter:
                best_fitness_iter = fitness
            if fitness < args.danger_threshold:
                current_hazardous.append(g['global_best_position'].copy())
                
        avg_fitness_iter = total_fitness_iter / pso.num_subgroups
        
        hazardous_points.extend(current_hazardous)
        
        # 计算当前的 NCS (Number of Critical Scenarios) (unique niches)
        current_ncs = 0
        if len(hazardous_points) > 0:
            hp_array = np.array(hazardous_points)
            fv_array = np.array([args.danger_threshold - 0.01]*len(hp_array)) # dummy fitness for filtering
            seeds_info = select_representative_seeds(hp_array, fv_array, niche_radius=args.niche_radius)
            current_ncs = len(seeds_info['representative'])
            
        # 记录 metrics (区分在线/离线预算)
        if args.use_carla:
            sim_budget = env.real_simulation_count
        else:
            sim_budget = env.surrogate_call_count
            
        budget_history.append(sim_budget)
        risk_score_history.append(-best_fitness_iter) # Max Risk score is negative fitness
        avg_risk_score_history.append(-avg_fitness_iter) # Avg Risk score
        ncs_history.append(current_ncs)
        
        if current_ncs > 0 and time_to_first_failure == -1:
            time_to_first_failure = sim_budget
            print(f"  [!] First failure found at simulation budget: {time_to_first_failure}")
        
        if (iteration + 1) % 100 == 0:
            best_fitness = min(g['global_best_fitness'] for g in pso.subgroups)
            print(f"  Iter {iteration+1}/{args.iterations} | Budget: {sim_budget} | Best Fitness: {best_fitness:.4f} | NCS: {current_ncs}")
    
    hazardous_points = np.array(hazardous_points) if hazardous_points else np.array([])
    search_time = time.time() - start_time
    
    print(f"\n  Search time: {search_time:.2f}s")
    print(f"  Raw hazardous points: {len(hazardous_points)}")
    print(f"  Time to First Failure (Budget): {time_to_first_failure}")
    
    if len(hazardous_points) == 0:
        print("\nNo hazardous points found. Consider lowering danger_threshold.")
        
        # 保存无结果但也记录了轨迹
        history_data = {
            'budget_history': np.array(budget_history),
            'risk_score_history': np.array(risk_score_history),
            'avg_risk_score_history': np.array(avg_risk_score_history),
            'ncs_history': np.array(ncs_history),
            'time_to_first_failure': time_to_first_failure,
            'search_time': search_time,
            'args': vars(args)
        }
        with open(os.path.join(args.output_dir, 'search_results.pkl'), 'wb') as f:
            pickle.dump(history_data, f)
        sio.savemat(os.path.join(args.output_dir, 'search_results.mat'), 
                   {k: v if not isinstance(v, dict) else str(v) for k,v in history_data.items()})
        return
    
    # 筛选种子
    print("\n[4] Filtering seeds...")
    # fitness_values, _ = env.evaluate(hazardous_points)
    # 取巧：为了不重复调用评价引发多余的 budget 计算，直接假定 fitness = gp_predict
    x_tensor = torch.tensor(hazardous_points, dtype=torch.float32).to("cuda")
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        fitness_values = -env.gp_likelihood(env.gp_model(x_tensor)).mean.cpu().numpy()
        
    seeds_info = select_representative_seeds(
        hazardous_points, fitness_values, 
        niche_radius=args.niche_radius, 
        niche_capacity=args.niche_capacity,
        top_k_dangerous=args.top_k
    )

    history_data = {
        'budget_history': np.array(budget_history),
        'risk_score_history': np.array(risk_score_history),
        'avg_risk_score_history': np.array(avg_risk_score_history),
        'ncs_history': np.array(ncs_history),
        'time_to_first_failure': time_to_first_failure,
        'search_time': search_time,
        'representative_seeds': seeds_info['representative'],
        'representative_fitness': seeds_info['representative_fitness'],
        'args': vars(args)
    }

    # 保存结果
    with open(os.path.join(args.output_dir, 'search_results.pkl'), 'wb') as f:
        pickle.dump(history_data, f)
        
    sio.savemat(os.path.join(args.output_dir, 'search_results.mat'), 
               {k: v if not isinstance(v, dict) else str(v) for k,v in history_data.items()})
               
    # 绘制基础图
    plot_seeds(seeds_info['representative'], seeds_info['representative_fitness'], 
               "RL-PSO Representative Seeds", os.path.join(args.output_dir, 'seeds_distribution.png'))
               
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(budget_history, risk_score_history, 'b-')
    plt.xlabel('Simulation Budget')
    plt.ylabel('Risk Score')
    plt.title('Risk Score vs Budget')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(budget_history, ncs_history, 'r-')
    plt.xlabel('Simulation Budget')
    plt.ylabel('NCS (Critical Scenarios)')
    plt.title('NCS vs Budget')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'metrics_convergence.png'), dpi=300)
    plt.close()
    
    print(f" Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
