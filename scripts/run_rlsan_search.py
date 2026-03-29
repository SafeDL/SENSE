"""
Reinforcement Learning-based Surrogate-Assisted Niching for Automated Driving Testing (Hybrid Mode)
(1) 初始化 Safebench CarlaRunner, 执行真实测试以获取结果
(2) 使用统一的 ADScenarioEnv 和 RLAgent 进行强化学习驱动的小生境自适应搜索
(3) 当预测不确定性较大时调用真实Carla仿真，否则代理模型输出 
"""
import pickle
import os
import os.path as osp
import torch
import numpy as np
import argparse
import time
import random

from safebench.util.run_util import load_config
from safebench.util.torch_util import set_torch_variable
from safebench.carla_runner_simple import CarlaRunner

from rlsan.src.RLSearch.utils import set_ieee_style, load_surrogate_model
from rlsan.src.RLSearch.envs.ad_scenario_env import ADScenarioEnv
from rlsan.src.RLSearch.optimizer.niche_pso import StateAwareNichePSO
from rlsan.src.RLSearch.deploy_ad_search import select_representative_seeds

# RL Agent imports
from rlsan.src.RLSearch.rl_core.dqn_agent import DoubleDQNAgent
from rlsan.src.RLSearch.rl_core.ppo_agent import PPOAgent
from rlsan.src.RLSearch.rl_core.sac_agent import SACAgent
from rlsan.src.RLSearch.rl_core.ddpg_agent import DDPGAgent
from rlsan.src.RLSearch.rl_core.td3_agent import TD3Agent
from rlsan.src.RLSearch.rl_core.trpo_agent import TRPOAgent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='rlsan')
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__)))))
    parser.add_argument('--auto_ego', type=bool, default=False)
    parser.add_argument('--max_episode_step', type=int, default=2000)
    parser.add_argument('--agent_cfg', nargs='+', type=str, default='behavior.yaml')
    parser.add_argument('--scenario_cfg', nargs='+', type=str, default='standard.yaml')

    parser.add_argument('--seed', '-s', type=int, default=42)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')   

    parser.add_argument('--num_scenario', '-ns', type=int, default=1, help='num of scenarios we run in one episode')
    parser.add_argument('--save_video', type=bool, default=False)
    parser.add_argument('--render', type=bool, default=True)
    parser.add_argument('--frame_skip', '-fs', type=int, default=1, help='skip of frame in each step')
    parser.add_argument('--port', type=int, default=2004, help='port to communicate with carla')
    parser.add_argument('--tm_port', type=int, default=8004, help='traffic manager port')
    parser.add_argument('--fixed_delta_seconds', type=float, default=0.1)
    
    # RLSearch 参数
    parser.add_argument('--rl_algo', type=str, default='sac', choices=['dqn', 'ppo', 'sac', 'ddpg', 'td3', 'trpo', 'none'])
    parser.add_argument('--pretrained', type=str, default='../rlsan/results/s1exp/sac_model_final.pth')
    parser.add_argument('--num_iterations', type=int, default=1000)
    parser.add_argument('--num_particles', type=int, default=2000)
    parser.add_argument('--num_subgroups', type=int, default=10)
    parser.add_argument('--danger_threshold', type=float, default=-0.3)
    parser.add_argument('--uncertainty_threshold', type=float, default=0.00217)
    parser.add_argument('--niche_radius', type=float, default=0.05)
    parser.add_argument('--dim', type=int, default=3)

    args = parser.parse_args()
    args_dict = vars(args)

    # 设置默认的输出目录（如果未指定）
    args.output_dir = osp.join(args.ROOT_DIR, 'log')

    os.makedirs(args.output_dir, exist_ok=True)

    set_torch_variable(args.device)
    torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    set_ieee_style()

    # 加载配置
    agent_config_path = osp.join(args.ROOT_DIR, 'safebench/agent/config', args.agent_cfg[0] if isinstance(args.agent_cfg, list) else args.agent_cfg)
    agent_config = load_config(agent_config_path)
    agent_config.update(args_dict)

    scenario_config_path = osp.join(args.ROOT_DIR, 'rlsan/config', args.scenario_cfg[0] if isinstance(args.scenario_cfg, list) else args.scenario_cfg)
    scenario_config = load_config(scenario_config_path)
    scenario_config.update(args_dict)

    print(f"\n[0] Output Directory: {osp.abspath(args.output_dir)}")
    print("\n[1] Initializing Carla Runner...")
    runner = CarlaRunner(agent_config, scenario_config, step_by_step=True)

    print("\n[2] Loading Surrogate Model...")
    model_path = r'/home/hp/SENSE/rlsan/results/s1exp/surrogate_model_1000.pkl'
    try:
        gp_model, gp_likelihood = load_surrogate_model(model_path)
        print("    Global GP Model loaded successfully.")
    except Exception as e:
        print(f"    Failed to load surrogate model: {model_path}. Error: {e}")
        exit(1)

    # 包装到 ADScenarioEnv
    env = ADScenarioEnv(
        gp_model=gp_model, gp_likelihood=gp_likelihood,
        runner=runner, dim=args.dim, bounds=(-1.0, 1.0),
        uncertainty_threshold=args.uncertainty_threshold
    )
    
    print(f"\n[3] Loading RL Agent ({args.rl_algo})...")
    agent = None
    if args.rl_algo != 'none':
        try:
            agent_dict = {
                'dqn': DoubleDQNAgent, 'ppo': PPOAgent, 'sac': SACAgent,
                'ddpg': DDPGAgent, 'td3': TD3Agent, 'trpo': TRPOAgent
            }
            agent = agent_dict[args.rl_algo].load(args.pretrained, freeze=True)
            print("    Agent loaded successfully.")
        except Exception as e:
            print(f"    Warning: Could not load agent '{args.rl_algo}': {e}. Using pure PSO.")
    
    print("\n[4] Initializing Optimizer & Running Search...")
    pso = StateAwareNichePSO(
        env=env, agent=agent,
        num_particles=args.num_particles,
        num_subgroups=args.num_subgroups,
        max_iterations=args.num_iterations,
        enable_restart=True,
        restart_patience=5
    )
    pso.init_subgroups(reset_trackers=True)

    fdc_curve_raw = []  # 基于原始点数量的FDC曲线
    fdc_curve_representative = []  # 基于代表性点数量的FDC曲线
    n_50_raw = -1  # 发现前 50 个原始失效用例所需的步数
    n_50_representative = -1  # 发现前 50 个代表性失效用例所需的步数
    start_time = time.time()

    for iteration in range(args.num_iterations):
        pso.run_one_iteration()

        # 直接使用PSO内部累积收集的所有危险粒子（而非仅子群最佳点）
        # 这样能获得更多危险场景，与GA方法一致
        current_hazardous_points = list(pso.global_hazardous_pool)

        # 获取当前迭代的最佳适应度（用于日志输出）
        best_fitness_iter = float('inf')
        for g in pso.subgroups:
            if g['global_best_fitness'] < best_fitness_iter:
                best_fitness_iter = g['global_best_fitness']

        # 原始失效点数量
        raw_failures_count = len(current_hazardous_points)

        # 计算当前代表性点数量 (Number of Critical Scenarios) - 去重后的代表性点
        current_ncs = 0
        if len(current_hazardous_points) > 0:
            hp_array = np.array(current_hazardous_points)
            fv_array = np.array([args.danger_threshold - 0.01]*len(hp_array))
            seeds_info = select_representative_seeds(hp_array, fv_array, niche_radius=args.niche_radius)
            current_ncs = len(seeds_info['representative'])

        search_budget = env.evaluation_count  # 总搜索次数（代理模型 + 真实CARLA）

        # 记录两条 FDC 曲线数据点
        fdc_curve_raw.append((search_budget, raw_failures_count))
        fdc_curve_representative.append((search_budget, current_ncs))

        # 记录发现前 50 个原始失效用例的时刻
        if raw_failures_count >= 50 and n_50_raw == -1:
            n_50_raw = search_budget

        # 记录发现前 50 个代表性失效用例的时刻
        if current_ncs >= 50 and n_50_representative == -1:
            n_50_representative = search_budget

        if (iteration + 1) % 10 == 0:
            print(f"    Iter {iteration+1}/{args.num_iterations} | Search Budget: {search_budget} | Best Risk: {-best_fitness_iter:.4f} | Raw Failures: {raw_failures_count} | Representative: {current_ncs}")

    end_time = time.time()
    search_time = end_time - start_time

    # 从PSO的global_hazardous_pool中取最终所有危险点
    hazardous_points = list(pso.global_hazardous_pool)

    # 计算两条 FDC 曲线的 AUC
    auc_fdc_raw = 0.0
    if len(fdc_curve_raw) > 1:
        budgets = np.array([p[0] for p in fdc_curve_raw])
        failures = np.array([p[1] for p in fdc_curve_raw])
        auc_fdc_raw = np.trapz(failures, budgets)

    auc_fdc_representative = 0.0
    if len(fdc_curve_representative) > 1:
        budgets = np.array([p[0] for p in fdc_curve_representative])
        failures = np.array([p[1] for p in fdc_curve_representative])
        auc_fdc_representative = np.trapz(failures, budgets)

    # 计算最终的代表性点数（去重后）
    final_ncs = 0
    if len(hazardous_points) > 0:
        hp_array = np.array(hazardous_points)
        fv_array = np.array([args.danger_threshold - 0.01]*len(hp_array))
        seeds_info = select_representative_seeds(hp_array, fv_array, niche_radius=args.niche_radius)
        final_ncs = len(seeds_info['representative'])

    hazardous_points = np.array(hazardous_points) if hazardous_points else np.array([])
    print(f"\nOptimization completed in {search_time:.2f} seconds.")
    print(f"Total Unique Points Evaluated: {env.evaluation_count}")
    print(f"  - Surrogate Predictions (Final): {env.surrogate_final_count}")
    print(f"  - Real Simulations (Final): {env.real_simulation_final_count}")
    print(f"  - Total Surrogate Calls (incl. replaced): {env.surrogate_call_count}")
    print(f"Raw Failures Found: {len(hazardous_points)}")
    print(f"Representative Failures (After Deduplication): {final_ncs}")

    print(f"\n--- Performance Metrics (Raw Failures) ---")
    print(f"AUC-FDC (Raw): {auc_fdc_raw:.4e}")
    if n_50_raw == -1:
        print(f"N_50 (Raw): Not reached (found {len(hazardous_points)}/50)")
    else:
        print(f"N_50 (Raw): {n_50_raw}")

    print(f"\n--- Performance Metrics (Representative Failures) ---")
    print(f"AUC-FDC (Representative): {auc_fdc_representative:.4e}")
    if n_50_representative == -1:
        print(f"N_50 (Representative): Not reached (found {final_ncs}/50)")
    else:
        print(f"N_50 (Representative): {n_50_representative}")

    # 仅保存危险场景点到pkl文件
    results = {
        'hazardous_points': hazardous_points,
        'raw_failures_count': len(hazardous_points),
        'representative_failures_count': final_ncs,
        'total_evaluations': env.evaluation_count,
        'surrogate_final_count': env.surrogate_final_count,
        'real_simulations': env.real_simulation_final_count,
        'surrogate_calls': env.surrogate_call_count,
        'search_time': search_time,
        'auc_fdc_raw': auc_fdc_raw,
        'n_50_raw': n_50_raw,
        'fdc_curve_raw': fdc_curve_raw,
        'auc_fdc_representative': auc_fdc_representative,
        'n_50_representative': n_50_representative,
        'fdc_curve_representative': fdc_curve_representative,
    }

    with open(osp.join(args.output_dir, 'search_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    # 规范化路径（移除 ../ 等相对路径符号）
    output_path = osp.abspath(osp.join(args.output_dir, 'search_results.pkl'))
    print(f"\nResults saved to {output_path}")

