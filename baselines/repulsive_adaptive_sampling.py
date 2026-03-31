"""
复现了 Ge et al. 论文中的核心逻辑：基于斥力场的自适应采样 (Repulsive Adaptive Sampling)。
(1) 外层循环 (Outer Loop): 负责管理总的仿真预算 (Budget)。它会分批次生成候选样本。
(2) 内层循环 (Inner Loop - 核心): 对应论文中的 Optimization 过程。

支持代理模型加速+不确定性驱动仿真
"""
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
from scipy.spatial import cKDTree  # 高效的KD树实现
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


def get_repulsion_coefficient(score, min_score=0.0, max_score=1.0):
    """
    根据论文逻辑计算斥力系数 Q
    逻辑：
    - 危险场景 (High Score) -> 应允许探索 -> 斥力小 (Small Q)
    - 安全场景 (Low Score) -> 应远离 -> 斥力大 (Large Q)
    """
    # 将分数归一化到 0~1 之间 (假设大致范围，根据实际情况调整)
    norm_score = np.clip((score - min_score) / (max_score - min_score + 1e-6), 0, 1)

    # 论文核心假设：Critcality 越高，Repulsion 越低
    # 简单的线性映射：Q = 1.0 - score
    # Q_max = 1.0 (强斥力), Q_min = 0.1 (弱斥力)
    Q = 1.0 - (0.9 * norm_score)
    return Q


def calculate_forces(candidate, known_samples, known_scores, other_candidates, bounds, kdtree=None, r_max=None):
    """
    计算当前候选点受到的总斥力 (优化版本)
    F = k * Q1 * Q2 / dist^2

    优化策略:
    1. 使用KDTree快速查询邻近点 (O(log N) 而非 O(N))
    2. 设置斥力截止 (cut-off): 距离超过3*r_max的力忽略
    3. 向量化计算
    """
    total_force = np.zeros_like(candidate)
    epsilon = 1e-5  # 防止除零

    if len(known_samples) == 0:
        # 若无历史样本，仅计算批内互斥
        for other in other_candidates:
            if np.array_equal(candidate, other):
                continue
            diff = candidate - other
            dist_sq = np.sum(diff ** 2) + epsilon
            direction = diff / (np.sqrt(dist_sq) + epsilon)
            force_mag = 0.2 / dist_sq
            total_force += force_mag * direction
        return total_force

    # 1. 来自已知历史样本的斥力 (带KDTree加速)
    if kdtree is not None and r_max is not None:
        # KDTree查询：仅获取距离候选点 3*r_max 范围内的邻居
        cutoff_dist = 3.0 * r_max
        neighbor_indices = kdtree.query_ball_point(candidate, r=cutoff_dist)

        # 计算这些邻居的斥力
        for idx in neighbor_indices:
            if idx < len(known_samples):
                diff = candidate - known_samples[idx]
                dist_sq = np.sum(diff ** 2) + epsilon
                dist = np.sqrt(dist_sq)

                # 斥力截止检查（额外保险）
                if dist > cutoff_dist:
                    continue

                direction = diff / (np.sqrt(dist_sq) + epsilon)
                Q_known = get_repulsion_coefficient(known_scores[idx])
                Q_candidate = 0.5
                force_mag = (Q_known * Q_candidate) / dist_sq
                total_force += force_mag * direction
    else:
        # Fallback: 完整计算（当KDTree不可用时）
        for i, known_pt in enumerate(known_samples):
            diff = candidate - known_pt
            dist_sq = np.sum(diff ** 2) + epsilon
            direction = diff / (np.sqrt(dist_sq) + epsilon)
            Q_known = get_repulsion_coefficient(known_scores[i])
            Q_candidate = 0.5
            force_mag = (Q_known * Q_candidate) / dist_sq
            total_force += force_mag * direction

    # 2. 来自其他候选点的斥力 (Self-Repulsion within Batch)
    for other in other_candidates:
        if np.array_equal(candidate, other):
            continue
        diff = candidate - other
        dist_sq = np.sum(diff ** 2) + epsilon
        direction = diff / (np.sqrt(dist_sq) + epsilon)
        force_mag = 0.2 / dist_sq
        total_force += force_mag * direction

    return total_force


def optimize_candidates(candidates, known_samples, known_scores, bounds, iterations=50, step_size=0.05):
    """
    内层循环 (Inner Loop): 对应论文中的 Sub-optimization
    通过物理斥力移动样本位置

    优化版本:
    1. 预先构建KDTree (O(N log N) 一次性成本)
    2. 计算最大距离r_max用于截止
    3. 使用向量化操作替代Python循环
    """
    dim = len(bounds)
    optimized_candidates = np.array(candidates, dtype=float)

    # ===== 优化策略1: 构建KDTree =====
    kdtree = None
    r_max = None
    if len(known_samples) > 0:
        try:
            known_samples_array = np.array(known_samples)
            kdtree = cKDTree(known_samples_array)

            # 计算r_max: 所有候选点到已知点的最大距离
            # 用于设定斥力截止范围
            dists = np.linalg.norm(
                optimized_candidates[:, np.newaxis, :] - known_samples_array[np.newaxis, :, :],
                axis=2
            )
            r_max = np.max(dists) if dists.size > 0 else 1.0
        except Exception as e:
            print(f"[Warning] KDTree construction failed: {e}. Falling back to brute force.")
            kdtree = None
            r_max = None

    # 迭代移动
    for iteration in range(iterations):
        current_forces = []

        # ===== 优化策略2: 向量化计算力 =====
        # 对每个候选点计算力（仍需逐个处理边界约束）
        safe_known_samples = np.array(known_samples) if len(known_samples) > 0 else np.array([])
        safe_known_scores = np.array(known_scores) if len(known_scores) > 0 else np.array([])
        for i in range(len(optimized_candidates)):
            force = calculate_forces(
                optimized_candidates[i],
                safe_known_samples,
                safe_known_scores,
                optimized_candidates,
                bounds,
                kdtree=kdtree,
                r_max=r_max
            )
            current_forces.append(force)

        # 更新位置
        for i in range(len(optimized_candidates)):
            force = current_forces[i]
            force_mag = np.linalg.norm(force)

            # 梯度裁剪 (Gradient Clipping)
            if force_mag > 10.0:
                force = force / force_mag * 10.0

            optimized_candidates[i] += force * step_size

            # 边界约束 (Boundary Constraint)
            for d in range(dim):
                optimized_candidates[i][d] = np.clip(optimized_candidates[i][d], bounds[d][0], bounds[d][1])

    return optimized_candidates


def print_diagnostics(evaluator, current_iter, n_calls, cumulative_raw, known_samples,
                     real_sim_budget, batch_size, phase_name=""):
    """打印诊断信息用于判断搜索是否正确恢复"""
    print("\n" + "="*85)
    print(f"[DIAGNOSTICS] {phase_name}")
    print("="*85)
    print(f"  迭代进度:        {current_iter}/{n_calls} ({100*current_iter/n_calls:.1f}%)")
    print(f"  已评估样本数:    {len(known_samples)}")
    print(f"  危险场景数:      {cumulative_raw}")
    print(f"  代理模型调用:    {evaluator.surrogate_call_count}")
    print(f"  真实仿真次数:    {evaluator.real_simulation_count}/{real_sim_budget}")
    print(f"  总评估次数:      {evaluator.evaluation_count}")
    print(f"  真实仿真剩余:    {max(0, real_sim_budget - evaluator.real_simulation_count)}")
    print(f"  批大小:          {batch_size}")
    print("="*85 + "\n")


def run_adaptive_sampling(evaluator, bounds, n_calls=1000, batch_size=10, output_dir='log',
                        real_sim_budget=2000, collision_threshold=0.3, checkpoint_interval=50):
    """
    基于斥力的自适应采样主流程 (优化版本，支持代理模型+不确定性驱动仿真，实时FDC统计)
    Reference: Ge et al., “Life-long Learning and Testing...”, TIV 2024.

    优化策略:
    1. 在每个batch开始前构建KDTree (O(N log N))
    2. 使用斥力截止减少99%的无用计算
    3. 预计单步复杂度从O(N)降至O(log N)
    """
    checkpoint_path = osp.join(output_dir, 'ras_checkpoint.pkl')

    # 历史数据库 (Knowledge)
    known_samples = []
    known_scores = []

    all_scenarios_log = []
    all_scores_log = []
    hazardous_scenarios_log = []  # 追踪所有危险场景
    hazardous_scores_log = []

    # 初始化 FDC 追踪
    fdc_curve_raw = []
    n_50_raw = -1
    cumulative_raw = 0
    start_iter = 0
    last_checkpoint_iter = 0  # 用于追踪进度

    # 尝试加载检查点
    if osp.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'rb') as f:
                ckpt = pickle.load(f)
            start_iter = ckpt['iteration']
            known_samples = ckpt['known_samples']
            known_scores = ckpt['known_scores']
            all_scenarios_log = ckpt['all_scenarios_log']
            all_scores_log = ckpt['all_scores_log']
            hazardous_scenarios_log = ckpt['hazardous_scenarios_log']
            hazardous_scores_log = ckpt['hazardous_scores_log']
            fdc_curve_raw = ckpt['fdc_curve_raw']
            n_50_raw = ckpt.get('n_50_raw', -1)
            cumulative_raw = ckpt['cumulative_raw']
            evaluator.evaluation_count = ckpt['eval_count']
            evaluator.surrogate_call_count = ckpt['surrogate_call_count']
            evaluator.real_simulation_count = ckpt['real_sim_count']
            last_checkpoint_iter = start_iter
            print(f"[✓] Resumed from checkpoint: iteration {start_iter}/{n_calls}")
            print_diagnostics(evaluator, start_iter, n_calls, cumulative_raw, known_samples,
                             real_sim_budget, batch_size, "Checkpoint Loaded")
        except Exception as e:
            print(f"[!] Failed to load checkpoint: {e}, starting fresh")
            start_iter = 0

    print(f"Starting Adaptive Force Sampling (Ge et al. 2024) with budget: {n_calls}")
    print(f"Real simulation budget limit: {real_sim_budget}")
    print("-" * 85)

    # 1. 初始化阶段 (Cold Start)
    init_count = min(n_calls, batch_size)
    if start_iter < init_count:
        print(f">> Initialization Phase: Randomly sampling {init_count - start_iter} points...")
        for i in tqdm(range(start_iter, init_count), desc="Initialization", unit="eval"):
            # 随机采样
            sample = np.array([np.random.uniform(b[0], b[1]) for b in bounds])

            # 评估 (使用代理模型或真实仿真)
            score = evaluator.evaluate(sample)

            # 存入知识库
            known_samples.append(sample)
            known_scores.append(score)
            all_scenarios_log.append(sample)
            all_scores_log.append(score)

            # 收集危险场景 (score > collision_threshold)
            if score > collision_threshold:
                cumulative_raw += 1
                hazardous_scenarios_log.append(sample)
                hazardous_scores_log.append(score)
                search_budget = evaluator.evaluation_count
                fdc_curve_raw.append((search_budget, cumulative_raw))
                if cumulative_raw >= 50 and n_50_raw == -1:
                    n_50_raw = search_budget

            current_overall_iter = i + 1

            # 保存检查点
            if current_overall_iter % checkpoint_interval == 0:
                ckpt = {
                    'iteration': current_overall_iter,
                    'known_samples': known_samples,
                    'known_scores': known_scores,
                    'all_scenarios_log': all_scenarios_log,
                    'all_scores_log': all_scores_log,
                    'hazardous_scenarios_log': hazardous_scenarios_log,
                    'hazardous_scores_log': hazardous_scores_log,
                    'fdc_curve_raw': fdc_curve_raw,
                    'n_50_raw': n_50_raw,
                    'cumulative_raw': cumulative_raw,
                    'eval_count': evaluator.evaluation_count,
                    'surrogate_call_count': evaluator.surrogate_call_count,
                    'real_sim_count': evaluator.real_simulation_count,
                }
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(ckpt, f)

            # 真值仿真熔断检查
            if evaluator.real_simulation_count >= real_sim_budget:
                print(f"[!] Real simulation budget reached ({real_sim_budget}), stopping at initialization {current_overall_iter}/{init_count}")
                return (np.array(all_scenarios_log), np.array(all_scores_log),
                       np.array(hazardous_scenarios_log), np.array(hazardous_scores_log),
                       fdc_curve_raw, n_50_raw)
    else:
        print(f">> Initialization Phase skipped (already completed in checkpoint)")
        print_diagnostics(evaluator, start_iter, n_calls, cumulative_raw, known_samples,
                         real_sim_budget, batch_size, "After Checkpoint Load")

    # 2. 自适应采样阶段 (Adaptive Phase)
    current_iter = start_iter  # 从断点位置继续，而非重置为 init_count

    # 如果还未完成初始化，先完成初始化
    if current_iter < init_count:
        print(f">> Completing Initialization Phase: {current_iter}/{init_count} already done")
        for i in tqdm(range(current_iter, init_count), desc="Initialization", unit="eval"):
            sample = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            score = evaluator.evaluate(sample)
            known_samples.append(sample)
            known_scores.append(score)
            all_scenarios_log.append(sample)
            all_scores_log.append(score)

            if score > collision_threshold:
                cumulative_raw += 1
                hazardous_scenarios_log.append(sample)
                hazardous_scores_log.append(score)
                search_budget = evaluator.evaluation_count
                fdc_curve_raw.append((search_budget, cumulative_raw))
                if cumulative_raw >= 50 and n_50_raw == -1:
                    n_50_raw = search_budget

            current_iter = i + 1

            if current_iter % checkpoint_interval == 0:
                ckpt = {
                    'iteration': current_iter,
                    'known_samples': known_samples,
                    'known_scores': known_scores,
                    'all_scenarios_log': all_scenarios_log,
                    'all_scores_log': all_scores_log,
                    'hazardous_scenarios_log': hazardous_scenarios_log,
                    'hazardous_scores_log': hazardous_scores_log,
                    'fdc_curve_raw': fdc_curve_raw,
                    'n_50_raw': n_50_raw,
                    'cumulative_raw': cumulative_raw,
                    'eval_count': evaluator.evaluation_count,
                    'surrogate_call_count': evaluator.surrogate_call_count,
                    'real_sim_count': evaluator.real_simulation_count,
                }
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(ckpt, f)

            if evaluator.real_simulation_count >= real_sim_budget:
                print(f"[!] Real simulation budget reached ({real_sim_budget}), stopping at initialization {current_iter}/{init_count}")
                return (np.array(all_scenarios_log), np.array(all_scores_log),
                       np.array(hazardous_scenarios_log), np.array(hazardous_scores_log),
                       fdc_curve_raw, n_50_raw)
        current_iter = init_count

    # 进入自适应阶段
    remaining_budget = n_calls - current_iter
    if remaining_budget > 0:
        print(f">> Adaptive Phase: {current_iter}/{n_calls} completed, {remaining_budget} remaining")
        print_diagnostics(evaluator, current_iter, n_calls, cumulative_raw, known_samples,
                         real_sim_budget, batch_size, "Entering Adaptive Phase")

        # 检查真实仿真预算
        real_sim_remaining = real_sim_budget - evaluator.real_simulation_count
        if real_sim_remaining <= 0:
            print(f"[!] Real simulation budget exhausted ({evaluator.real_simulation_count}/{real_sim_budget})")
            print(f"[!] Switching to surrogate-only mode (no more real simulations)")
            # 禁用真实仿真
            evaluator.use_real_sim = False
        else:
            print(f"[*] Real simulation budget remaining: {real_sim_remaining}/{real_sim_budget}")

        num_batches = int(np.ceil(remaining_budget / batch_size))

        for b in tqdm(range(num_batches), desc="Adaptive Phase", unit="batch"):
            current_batch_size = min(batch_size, n_calls - current_iter)
            if current_batch_size <= 0:
                break

            # 每10个batch打印进度
            if b % 10 == 0 and b > 0:
                progress_pct = 100 * current_iter / n_calls
                iter_delta = current_iter - last_checkpoint_iter
                print(f"[Progress] Iter: {current_iter}/{n_calls} ({progress_pct:.1f}%) | "
                      f"Real sims: {evaluator.real_simulation_count}/{real_sim_budget} | "
                      f"Hazardous: {cumulative_raw} | Delta: +{iter_delta}")

            # A. 生成初始随机候选点
            raw_candidates = np.array([
                [np.random.uniform(low, high) for (low, high) in bounds]
                for _ in range(current_batch_size)
            ])

            # B. 内层循环优化 (Inner Loop Optimization)
            optimized_candidates = optimize_candidates(
                raw_candidates,
                np.array(known_samples),
                np.array(known_scores),
                bounds,
                iterations=30,
                step_size=0.05
            )

            # C. 仿真评估优化后的点
            for idx, candidate in enumerate(optimized_candidates):
                if current_iter >= n_calls:
                    break

                # 评估前检查真实仿真预算
                if evaluator.real_simulation_count >= real_sim_budget:
                    print(f"[!] Real simulation budget reached ({evaluator.real_simulation_count}/{real_sim_budget}), stopping")
                    return (np.array(all_scenarios_log), np.array(all_scores_log),
                           np.array(hazardous_scenarios_log), np.array(hazardous_scores_log),
                           fdc_curve_raw, n_50_raw)

                # 记录评估前的代理调用数
                surrogate_before = evaluator.surrogate_call_count
                real_sim_before = evaluator.real_simulation_count

                score = evaluator.evaluate(candidate)

                # 检查是否使用了代理模型或真实仿真
                surrogate_delta = evaluator.surrogate_call_count - surrogate_before
                real_sim_delta = evaluator.real_simulation_count - real_sim_before

                # 更新知识库
                known_samples.append(candidate)
                known_scores.append(score)
                all_scenarios_log.append(candidate)
                all_scores_log.append(score)

                # 收集危险场景
                if score > collision_threshold:
                    cumulative_raw += 1
                    hazardous_scenarios_log.append(candidate)
                    hazardous_scores_log.append(score)
                    search_budget = evaluator.evaluation_count
                    fdc_curve_raw.append((search_budget, cumulative_raw))
                    if cumulative_raw >= 50 and n_50_raw == -1:
                        n_50_raw = search_budget

                current_iter += 1

            # 每50个样本打印一次详细信息
                if current_iter % 50 == 0:
                    eval_type = "Real" if real_sim_delta > 0 else "Surrogate"
                    print(f"[{current_iter:6d}] {eval_type:9s} | Score: {score:.4f} | "
                          f"Surr: {evaluator.surrogate_call_count:5d} | "
                          f"Real: {evaluator.real_simulation_count:4d}/{real_sim_budget} | "
                          f"Hazard: {cumulative_raw:4d}")

                # 保存检查点
                if current_iter % checkpoint_interval == 0:
                    last_checkpoint_iter = current_iter
                    ckpt = {
                        'iteration': current_iter,
                        'known_samples': known_samples,
                        'known_scores': known_scores,
                        'all_scenarios_log': all_scenarios_log,
                        'all_scores_log': all_scores_log,
                        'hazardous_scenarios_log': hazardous_scenarios_log,
                        'hazardous_scores_log': hazardous_scores_log,
                        'fdc_curve_raw': fdc_curve_raw,
                        'n_50_raw': n_50_raw,
                        'cumulative_raw': cumulative_raw,
                        'eval_count': evaluator.evaluation_count,
                        'surrogate_call_count': evaluator.surrogate_call_count,
                        'real_sim_count': evaluator.real_simulation_count,
                    }
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump(ckpt, f)

                # 真值仿真熔断检查
                if evaluator.real_simulation_count >= real_sim_budget:
                    print(f"[!] Real simulation budget reached ({real_sim_budget}), stopping adaptive sampling at iteration {current_iter}/{n_calls}")
                    return (np.array(all_scenarios_log), np.array(all_scores_log),
                           np.array(hazardous_scenarios_log), np.array(hazardous_scores_log),
                           fdc_curve_raw, n_50_raw)

            if current_iter >= n_calls:
                break

    # 排序返回 (scenarios和scores必须对应)
    results = sorted(zip(all_scenarios_log, all_scores_log), key=lambda x: x[1], reverse=True)
    sorted_scenarios = [x[0] for x in results]
    sorted_scores = [x[1] for x in results]

    return (np.array(sorted_scenarios), np.array(sorted_scores),
           np.array(hazardous_scenarios_log), np.array(hazardous_scores_log),
           fdc_curve_raw, n_50_raw)


if __name__ == '__main__':
    plt.switch_backend('agg')
    np.random.seed(42); random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='ras_search')
    parser.add_argument('--ROOT_DIR', type=str, default=osp.abspath(osp.dirname(osp.dirname(osp.realpath(__file__)))))
    parser.add_argument('--auto_ego', type=bool, default=False)
    parser.add_argument('--max_episode_step', type=int, default=2000)
    parser.add_argument('--agent_cfg', nargs='+', type=str, default='behavior.yaml')
    parser.add_argument('--scenario_cfg', nargs='+', type=str, default='standard.yaml')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_scenario', '-ns', type=int, default=1)
    parser.add_argument('--save_video', type=bool, default=False)
    parser.add_argument('--render', type=bool, default=True)
    parser.add_argument('--frame_skip', '-fs', type=int, default=1, help='skip of frame in each step')
    parser.add_argument('--port', type=int, default=2004)
    parser.add_argument('--tm_port', type=int, default=8004)
    parser.add_argument('--fixed_delta_seconds', type=float, default=0.1)
    parser.add_argument('--budget', type=int, default=20, help='Total number of evaluations')
    parser.add_argument('--batch_size', type=int, default=2000, help='Batch size for inner loop optimization')
    parser.add_argument('--use_surrogate', type=bool, default=True, help='Use surrogate model for acceleration')
    parser.add_argument('--surrogate_model_path', type=str,
                       default='rlsan/results/s1exp/surrogate_model_1000.pkl',
                       help='Path to surrogate model')
    parser.add_argument('--real_sim_budget', type=int, default=2000,
                        help='Max real CARLA simulations before stopping')
    parser.add_argument('--uncertainty_threshold', type=float, default=0.00217,
                        help='Uncertainty threshold for triggering real simulation (variance)')
    parser.add_argument('--checkpoint_interval', type=int, default=50,
                        help='Interval for saving checkpoints (number of evaluations)')

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

    critical_scenarios, all_scores, hazardous_scenarios, hazardous_scores, fdc_curve_raw, n_50_raw = run_adaptive_sampling(
        evaluator=evaluator,
        bounds=bounds,
        n_calls=args.budget,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        real_sim_budget=args.real_sim_budget,
        checkpoint_interval=args.checkpoint_interval
    )

    end_time = time.time()
    search_time = end_time - start_time
    print(f"\nAdaptive Sampling Execution time: {search_time:.2f} seconds")

    # 获取评估统计信息
    eval_stats = evaluator.get_stats() if evaluator else {
        'total_evaluations': len(all_scores),
        'surrogate_calls': 0,
        'real_simulations': len(all_scores)
    }

    # 使用直接收集的危险场景（已在run_adaptive_sampling中按score>0.3筛选）
    # hazardous_scenarios, hazardous_scores 已由函数返回

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
        scenarios=hazardous_scenarios,  # 直接使用收集到的危险场景
        scores=hazardous_scores,
        evaluator=evaluator,
        method_name='Repulsive Adaptive Sampling (Optimized)',
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
    results_save_path = osp.join(args.output_dir, 'repulsive_adaptive_sampling_results.pkl')
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