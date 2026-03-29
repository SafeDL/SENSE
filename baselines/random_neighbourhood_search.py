import pickle
import scipy.io as sio
import os
import os.path as osp
import torch
import numpy as np
import argparse
import time
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from safebench.util.run_util import load_config
from safebench.util.torch_util import set_seed, set_torch_variable
from safebench.carla_runner_simple import CarlaRunner
from surrogate_utils import SurrogateModel, SurrogateEvaluator, prepare_baseline_results


def select_representative_seeds(hazardous_points: np.ndarray,
                                fitness_values: np.ndarray,
                                niche_radius: float = 0.05,
                                niche_capacity: int = 3) -> dict:
    """筛选最危险和最有代表性的危险种子 (与RLSAN实现一致)"""
    if len(hazardous_points) == 0:
        return {
            'representative': np.array([]),
            'representative_fitness': np.array([])
        }

    # 去重：精确去重
    _, unique_idx = np.unique(np.round(hazardous_points, decimals=4), axis=0, return_index=True)
    unique_points = hazardous_points[unique_idx]
    unique_fitness = fitness_values[unique_idx]

    # 基于小生境覆盖的代表性种子选择
    sorted_idx = np.argsort(unique_fitness)
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

    return {
        'representative': representative,
        'representative_fitness': representative_fitness
    }


def run_rnns(evaluator, bounds, n_calls=1000, exploration_rate=0.3, perturbation_scale=0.1, output_dir='log',
            real_sim_budget=2000, collision_threshold=0.3):
    """
    随机最近邻搜索 (Random Nearest Neighbor Search / Epsilon-Greedy Local Search)
    支持代理模型加速+不确定性驱动仿真，实时FDC统计

    Args:
        evaluator: SurrogateEvaluator 实例
        exploration_rate: 以多大概率进行纯随机探索 (Exploration)
        perturbation_scale: 在最佳点附近进行局部搜索时的扰动大小 (Exploitation)
        real_sim_budget: 真值仿真预算上限
        collision_threshold: 碰撞阈值
    """
    dim = len(bounds)
    critical_scenarios = []
    all_scores = []

    # 记录当前的全局最优解
    best_x_so_far = None
    best_score_so_far = -float('inf')

    # 初始化 FDC 追踪
    fdc_curve_raw = []
    n_50_raw = -1
    cumulative_raw = 0

    print(f"Starting RNNS with budget: {n_calls}")
    print(f"Params: Explore Rate={exploration_rate}, Perturb Scale={perturbation_scale}")
    print(f"Real simulation budget limit: {real_sim_budget}")
    print("-" * 75)

    for i in tqdm(range(n_calls), desc="RNNS Search", unit="eval"):
        # 策略决策：
        # 1. 如果是第一次迭代，或者随机数小于 exploration_rate -> 纯随机探索
        # 2. 否则 -> 在当前最优解附近进行局部扰动 (Exploitation)

        if best_x_so_far is None or random.random() < exploration_rate:
            # 模式 A: 全局随机探索
            next_x = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            sample_type = "Random"
        else:
            # 模式 B: 局部开发 (Local Search)
            # 在 best_x 基础上加高斯噪声
            noise = np.random.normal(0, perturbation_scale, size=dim)
            candidate = best_x_so_far + noise

            # 必须 Clip 到边界内
            next_x = []
            for d in range(dim):
                clipped_val = np.clip(candidate[d], bounds[d][0], bounds[d][1])
                next_x.append(clipped_val)
            next_x = np.array(next_x)
            sample_type = "Local"

        # 评估 (使用代理模型或真实仿真)
        current_score = evaluator.evaluate(next_x)

        # 更新全局最优
        if current_score > best_score_so_far:
            best_score_so_far = current_score
            best_x_so_far = next_x

        # 记录数据
        critical_scenarios.append(next_x)
        all_scores.append(current_score)

        # 动态FDC统计 - 判断是否危险
        if current_score > collision_threshold:
            cumulative_raw += 1
            search_budget = evaluator.evaluation_count
            fdc_curve_raw.append((search_budget, cumulative_raw))
            if cumulative_raw >= 50 and n_50_raw == -1:
                n_50_raw = search_budget

        # 真值仿真熔断检查
        if evaluator.real_simulation_count >= real_sim_budget:
            print(f"[!] Real simulation budget reached ({real_sim_budget}), stopping RNNS at iteration {i+1}/{n_calls}")
            break

    # 排序返回 (scenarios和scores必须对应)
    results = sorted(zip(critical_scenarios, all_scores), key=lambda x: x[1], reverse=True)
    sorted_scenarios = [x[0] for x in results]
    sorted_scores = [x[1] for x in results]

    return np.array(sorted_scenarios), np.array(sorted_scores), fdc_curve_raw, n_50_raw


if __name__ == '__main__':
    plt.switch_backend('agg')
    np.random.seed(42); random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='rnns_search')
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__)))))
    parser.add_argument('--auto_ego', type=bool, default=False)
    parser.add_argument('--max_episode_step', type=int, default=2000)
    parser.add_argument('--agent_cfg', nargs='+', type=str, default='behavior.yaml')
    parser.add_argument('--scenario_cfg', nargs='+', type=str, default='standard.yaml')
    parser.add_argument('--seed', '-s', type=int, default=42)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_scenario', '-ns', type=int, default=1)
    parser.add_argument('--save_video', type=bool, default=False)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--frame_skip', '-fs', type=int, default=1, help='skip of frame in each step')
    parser.add_argument('--port', type=int, default=2004)
    parser.add_argument('--tm_port', type=int, default=8004)
    parser.add_argument('--fixed_delta_seconds', type=float, default=0.1)
    parser.add_argument('--budget', type=int, default=2000, help='Total number of evaluations')
    parser.add_argument('--rnns_explore_rate', type=float, default=0.3, help='Probability of random exploration')
    parser.add_argument('--rnns_perturb', type=float, default=0.1, help='Perturbation scale for local search')
    parser.add_argument('--use_surrogate', type=bool, default=False, help='Use surrogate model for acceleration')
    parser.add_argument('--surrogate_model_path', type=str,
                       default='rlsan/results/s1exp/surrogate_model_1000.pkl',
                       help='Path to surrogate model')
    parser.add_argument('--real_sim_budget', type=int, default=2000,
                        help='Max real CARLA simulations before stopping')
    parser.add_argument('--uncertainty_threshold', type=float, default=0.00,
                        help='Uncertainty threshold for triggering real simulation (variance)')

    args = parser.parse_args()
    args_dict = vars(args)

    # 设置输出目录
    args.output_dir = osp.join(args.ROOT_DIR, 'log', 'baselines')
    os.makedirs(args.output_dir, exist_ok=True)

    set_torch_variable(args.device)
    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    agent_config_path = osp.join(args.ROOT_DIR, 'safebench/agent/config', args.agent_cfg)
    agent_config = load_config(agent_config_path)
    agent_config.update(args_dict)

    scenario_config_path = osp.join(args.ROOT_DIR, 'rlsan/config', args.scenario_cfg)
    scenario_config = load_config(scenario_config_path)
    scenario_config.update(args_dict)

    runner = CarlaRunner(agent_config, scenario_config, step_by_step=True)
    bounds = [(-1, 1), (-1, 1), (-1, 1)]

    # 初始化评估器 (使用代理模型加速或真实CARLA)
    if args.use_surrogate:
        # 规范化模型路径
        surrogate_path = osp.abspath(osp.join(args.ROOT_DIR, args.surrogate_model_path))
        if osp.exists(surrogate_path):
            print(f"[*] Loading surrogate model from: {surrogate_path}")
            surrogate = SurrogateModel(surrogate_path)
            evaluator = SurrogateEvaluator(surrogate, real_runner=runner, use_real_sim=True,
                                          uncertainty_threshold=args.uncertainty_threshold)
            print(f"[✓] Surrogate evaluator initialized (with uncertainty-driven real sim)")
        else:
            print(f"[!] Surrogate model not found at {surrogate_path}")
            print(f"[*] Falling back to real CARLA simulations")
            surrogate = SurrogateModel(None)  # Dummy model
            evaluator = SurrogateEvaluator(surrogate, real_runner=runner, use_real_sim=True)
    else:
        print(f"[*] Using real CARLA simulations (surrogate disabled)")
        surrogate = SurrogateModel(None)  # Dummy model
        evaluator = SurrogateEvaluator(surrogate, real_runner=runner, use_real_sim=True)

    start_time = time.time()

    critical_scenarios, all_scores, fdc_curve_raw, n_50_raw = run_rnns(
        evaluator=evaluator,
        bounds=bounds,
        n_calls=args.budget,
        exploration_rate=args.rnns_explore_rate,
        perturbation_scale=args.rnns_perturb,
        output_dir=args.output_dir,
        real_sim_budget=args.real_sim_budget
    )

    end_time = time.time()
    search_time = end_time - start_time
    print(f"RNNS Execution time: {search_time:.2f} seconds")

    # 获取评估统计信息
    eval_stats = evaluator.get_stats() if evaluator else {
        'total_evaluations': args.budget,
        'surrogate_calls': 0,
        'real_simulations': args.budget
    }

    # 提取危险样本 (用于去重和统计)
    collision_mask = all_scores > 0.3
    hazardous_scenarios = critical_scenarios[collision_mask]
    hazardous_scores = all_scores[collision_mask]

    # 进行去重处理 (与RLSAN一致)
    if len(hazardous_scenarios) > 0:
        rep_seeds_info = select_representative_seeds(hazardous_scenarios, -hazardous_scores, niche_radius=0.05)
        representative_scenarios = rep_seeds_info['representative']
    else:
        representative_scenarios = np.array([]).reshape(0, 3)

    print(f"\n[*] Raw hazardous scenarios: {len(hazardous_scenarios)}")
    print(f"[*] Representative hazardous scenarios: {len(representative_scenarios)}")

    # 格式化结果为RLSAN兼容格式
    baseline_results = prepare_baseline_results(
        scenarios=hazardous_scenarios,
        scores=hazardous_scores,
        evaluator=evaluator,
        method_name='Random Neighbourhood Search',
        search_time=search_time,
        grid_x_data=None,
        grid_y_data=None,
        all_scores_history=np.array(all_scores)
    )

    # 添加代表性失效信息
    baseline_results['representative_failures_count'] = len(representative_scenarios)

    # 添加动态FDC数据
    baseline_results['fdc_curve_raw'] = fdc_curve_raw
    baseline_results['n_50_raw'] = n_50_raw

    # 计算 AUC 使用 FDC 数据
    if len(fdc_curve_raw) > 1:
        budgets = np.array([p[0] for p in fdc_curve_raw])
        failures = np.array([p[1] for p in fdc_curve_raw])
        baseline_results['auc_fdc_raw'] = float(np.trapz(failures, budgets))
    else:
        baseline_results['auc_fdc_raw'] = 0.0

    # 保存结果
    results_save_path = osp.join(args.output_dir, 'random_neighbourhood_search_results.pkl')
    with open(results_save_path, 'wb') as f:
        pickle.dump(baseline_results, f)
    print(f"[✓] Results saved to: {results_save_path}")

    # 打印统计信息
    print(f"\n[Summary]")
    print(f"Total Unique Points Evaluated: {eval_stats['total_evaluations']}")
    print(f"  - Surrogate Predictions (Final): {eval_stats['surrogate_final']}")
    print(f"  - Real Simulations: {eval_stats['real_simulations']}")
    print(f"  - Total Surrogate Calls (incl. replaced): {eval_stats['surrogate_calls']}")
    print(f"Raw Hazardous Found: {baseline_results['raw_failures_count']}")
    print(f"Representative Hazardous: {baseline_results['representative_failures_count']}")
    print(f"Search Time: {search_time:.2f}s")

    print(f"\n[FDC Metrics - Raw Failures]")
    print(f"AUC-FDC (Raw): {baseline_results['auc_fdc_raw']:.4e}")
    if n_50_raw == -1:
        print(f"N_50 (Raw): Not reached (found {baseline_results['raw_failures_count']}/50)")
    else:
        print(f"N_50 (Raw): {n_50_raw}")