# -*- coding: utf-8 -*-
"""
遗传算法搜索危险场景 (支持代理模型加速+不确定性驱动仿真)
"""
import pickle
import scipy.io as sio
import os
import os.path as osp
import torch
import numpy as np
import argparse
import random
import time
from tqdm import tqdm
from safebench.util.run_util import load_config
from safebench.util.torch_util import set_seed, set_torch_variable
from safebench.carla_runner_simple import CarlaRunner
from surrogate_utils import SurrogateModel, SurrogateEvaluator, prepare_baseline_results


def select_representative_seeds(hazardous_points: np.ndarray,
                                fitness_values: np.ndarray,
                                niche_radius: float = 0.05,
                                niche_capacity: int = 3) -> dict:
    """
    筛选最危险和最有代表性的危险种子 (与RLSAN实现一致)

    Args:
        hazardous_points: 危险点集合 (N, dim)
        fitness_values: 对应的适应度值 (N,)
        niche_radius: 小生境半径
        niche_capacity: 每个小生境容量

    Returns:
        dict 包含代表性种子及其适应度
    """
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


def evaluate_scenarios(scenarios, evaluator=None):
    """评估一组场景危险度，返回危险评分列表 (支持代理模型)"""
    scores = []
    for params in scenarios:
        score = evaluator.evaluate(params)
        scores.append(score)
    return np.array(scores).reshape(-1)


def genetic_algorithm(bounds, pop_size=50, generations=50, evaluator=None, mutation_rate=0.1, elite_fraction=0.2,
                     danger_threshold=0.3, real_sim_budget=2000):
    """
    经典遗传算法优化危险场景 (支持代理模型+不确定性驱动仿真，实时FDC统计)

    注意：与RLSAN一致，fitness = -score，最小化fitness
    即：最大化原始分数（危险度）
    """
    dim = len(bounds)

    # 初始化种群
    population = np.array([
        [np.random.uniform(low, high) for (low, high) in bounds]
        for _ in range(pop_size)
    ])

    # 仅存储危险样本，不存储所有历史 (节省内存)
    hazardous_scenarios = []
    hazardous_scores = []
    all_scores_log = []  # 用于FDC曲线

    # 初始化 FDC 追踪
    fdc_curve_raw = []
    n_50_raw = -1
    cumulative_raw = 0

    for gen in tqdm(range(generations), desc="GA Search", unit="gen"):
        # 评估适应度 (获得原始分数)
        raw_scores = evaluate_scenarios(population, evaluator)

        # 转换为fitness (与RLSAN一致：fitness = -score)
        fitness = -raw_scores
        all_scores_log.extend(raw_scores)

        # 保存本代的危险场景 (分数 > 危险阈值，即fitness < -危险阈值)
        collision_mask = raw_scores > danger_threshold
        gen_hazardous = population[collision_mask]
        gen_hazardous_scores = raw_scores[collision_mask]

        if len(gen_hazardous) > 0:
            hazardous_scenarios.extend(gen_hazardous.tolist())
            hazardous_scores.extend(gen_hazardous_scores.tolist())
            # 更新 FDC 统计
            for score in gen_hazardous_scores:
                cumulative_raw += 1
                search_budget = evaluator.evaluation_count
                fdc_curve_raw.append((search_budget, cumulative_raw))
                if cumulative_raw >= 50 and n_50_raw == -1:
                    n_50_raw = search_budget

        print(f"Generation {gen+1}/{generations}: best score = {raw_scores.max():.4f}, hazardous in this gen = {len(gen_hazardous)}")

        # 熔断检查：真值仿真预算达到上限
        if evaluator.real_simulation_count >= real_sim_budget:
            print(f"[!] Real simulation budget reached ({real_sim_budget}), stopping GA at generation {gen+1}/{generations}")
            break

        # 选择操作（基于fitness最小化）
        # 使用inverse ranking: 较小的fitness获得较高的选择概率
        # 这与PSO最小化目标一致
        fitness_inverted = -fitness  # 转回正值
        fitness_inverted = fitness_inverted - fitness_inverted.min() + 1e-8  # 确保都是正值
        probs = fitness_inverted / fitness_inverted.sum()

        selected = population[np.random.choice(pop_size, pop_size, p=probs)]

        # 交叉操作
        offspring = []
        for i in range(0, pop_size, 2):
            p1, p2 = selected[i], selected[(i+1) % pop_size]
            cross_point = random.randint(1, dim-1)
            child1 = np.concatenate([p1[:cross_point], p2[cross_point:]])
            child2 = np.concatenate([p2[:cross_point], p1[cross_point:]])
            offspring.append(child1)
            offspring.append(child2)
        offspring = np.array(offspring)

        # 变异操作
        for i in range(pop_size):
            if random.random() < mutation_rate:
                m_idx = random.randint(0, dim-1)
                low, high = bounds[m_idx]
                offspring[i][m_idx] = np.random.uniform(low, high)

        # 新种群 = 精英 + 后代
        # 精英是fitness最小的个体（最高分数的个体）
        elite_count = int(elite_fraction * pop_size)
        elite_idx = np.argsort(fitness)[:elite_count]  # 最小fitness的精英
        elites = population[elite_idx]
        population = np.vstack([elites, offspring[:pop_size - elite_count]])

    # 转换为数组
    hazardous_scenarios = np.array(hazardous_scenarios) if hazardous_scenarios else np.array([]).reshape(0, dim)
    hazardous_scores = np.array(hazardous_scores)

    return hazardous_scenarios, hazardous_scores, np.array(all_scores_log), fdc_curve_raw, n_50_raw


if __name__ == '__main__':
    # 设置随机种子
    np.random.seed(42); random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='ga_search')
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__)))))
    parser.add_argument('--auto_ego', type=bool, default=False)
    parser.add_argument('--max_episode_step', type=int, default=2000)
    parser.add_argument('--agent_cfg', nargs='+', type=str, default='behavior.yaml')
    parser.add_argument('--scenario_cfg', nargs='+', type=str, default='standard.yaml')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_scenario', '-ns', type=int, default=1, help='num of scenarios we run in one episode')
    parser.add_argument('--save_video', type=bool, default=False)
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--frame_skip', '-fs', type=int, default=1, help='skip of frame in each step')
    parser.add_argument('--port', type=int, default=2004, help='port to communicate with carla')
    parser.add_argument('--tm_port', type=int, default=8004, help='traffic manager port')
    parser.add_argument('--fixed_delta_seconds', type=float, default=0.1)
    parser.add_argument('--ga_pop_size', type=int, default=50, help='GA population size')
    parser.add_argument('--ga_generations', type=int, default=40, help='GA generations')
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

    # 解析和加载配置文件(主要是被测试的自动驾驶模型)
    set_torch_variable(args.device)
    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    # load agent config
    agent_config_path = osp.join(args.ROOT_DIR, 'safebench/agent/config', args.agent_cfg)
    agent_config = load_config(agent_config_path)
    agent_config.update(args_dict)

    # load scenario config
    scenario_config_path = osp.join(args.ROOT_DIR, 'rlsan/config', args.scenario_cfg)
    scenario_config = load_config(scenario_config_path)
    scenario_config.update(args_dict)

    runner = CarlaRunner(agent_config, scenario_config, step_by_step=True)
    bounds = [(-1, 1), (-1, 1), (-1, 1)]  # 场景参数范围

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

    # 计算GA参数使其接近总预算
    print(f"[*] GA params: pop_size={args.ga_pop_size}, generations={args.ga_generations}")

    hazardous_scenarios, hazardous_scores, all_scores, fdc_curve_raw, n_50_raw = genetic_algorithm(
        bounds,
        pop_size=args.ga_pop_size,
        generations=args.ga_generations,
        evaluator=evaluator,
        mutation_rate=0.1,
        elite_fraction=0.2,
        real_sim_budget=args.real_sim_budget
    )

    end_time = time.time()
    search_time = end_time - start_time
    print(f"GA Execution time: {search_time:.2f} seconds")

    # 获取评估统计信息
    eval_stats = evaluator.get_stats() if evaluator else {
        'total_evaluations': len(all_scores),
        'surrogate_calls': 0,
        'real_simulations': len(all_scores)
    }

    # 进行去重处理 (与RLSAN一致)
    if len(hazardous_scenarios) > 0:
        rep_seeds_info = select_representative_seeds(hazardous_scenarios, -hazardous_scores,
                                                     niche_radius=0.05)
        representative_scenarios = rep_seeds_info['representative']
    else:
        representative_scenarios = np.array([]).reshape(0, 3)

    print(f"\n[*] Raw hazardous scenarios: {len(hazardous_scenarios)}")
    print(f"[*] Representative hazardous scenarios: {len(representative_scenarios)}")

    # 格式化结果为RLSAN兼容格式
    baseline_results = prepare_baseline_results(
        scenarios=hazardous_scenarios,  # 仅包含危险样本
        scores=hazardous_scores if len(hazardous_scenarios) > 0 else np.array([]),
        evaluator=evaluator,
        method_name='Genetic Algorithm',
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
    results_save_path = osp.join(args.output_dir, 'genetic_algorithm_results.pkl')
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
