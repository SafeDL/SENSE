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

import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from safebench.util.run_util import load_config
from safebench.util.torch_util import set_seed, set_torch_variable
from safebench.carla_runner_simple import CarlaRunner
from surrogate_utils import SurrogateModel, SurrogateEvaluator, prepare_baseline_results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GPModel(gpytorch.models.ExactGP):
    """GPyTorch 高斯过程模型"""
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


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


def run_bayesian_optimization(evaluator, bounds, n_calls=500, random_state=42, output_dir='log',
                             real_sim_budget=2000, collision_threshold=0.3):
    """
    批量采样贝叶斯优化 (使用 GPyTorch 加速)
    """
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    critical_scenarios = []
    all_scores = []

    fdc_curve_raw = []
    n_50_raw = -1
    cumulative_raw = 0

    # 初始随机采样
    n_init = min(200, n_calls // 100)
    for _ in range(n_init):
        x = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        score = evaluator.evaluate(x)
        critical_scenarios.append(x)
        all_scores.append(score)

        if score > collision_threshold:
            cumulative_raw += 1
            fdc_curve_raw.append((evaluator.evaluation_count, cumulative_raw))
            if cumulative_raw >= 50 and n_50_raw == -1:
                n_50_raw = evaluator.evaluation_count

        if evaluator.real_simulation_count >= real_sim_budget:
            break

    if evaluator.real_simulation_count >= real_sim_budget:
        return np.array(critical_scenarios), np.array(all_scores), fdc_curve_raw, n_50_raw

    # 训练初始 GP 模型
    X_train = torch.tensor(np.array(critical_scenarios), dtype=torch.float32).to(device)
    Y_train = torch.tensor(np.array(all_scores), dtype=torch.float32).to(device)

    likelihood = GaussianLikelihood().to(device)
    model = GPModel(X_train, Y_train, likelihood).to(device)

    # 批量采样循环
    print(f"Starting Bayesian Optimization with budget: {n_calls}")
    print("-" * 60)

    batch_size = 200  # 每次采样 N 个点后再训练
    for i in tqdm(range(n_init, n_calls), desc="BO Search", unit="eval"):
        # 生成候选点
        candidates = np.array([np.random.uniform(b[0], b[1], size=1000) for b in bounds]).T
        X_cand = torch.tensor(candidates, dtype=torch.float32).to(device)

        # 批量预测
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = likelihood(model(X_cand))
            mean = posterior.mean.cpu().numpy()
            std = posterior.stddev.cpu().numpy()

        # UCB 采集函数：优先选择高不确定性的点（驱动真实仿真）
        # 这样 evaluator 会自动根据 uncertainty_threshold 决定是否调用真实 CARLA
        ucb = mean + 2.576 * std
        best_idx = np.argmax(ucb)
        next_x = candidates[best_idx]

        # 评估：evaluator 内部会根据 surrogate 不确定性自动调用真实仿真
        score = evaluator.evaluate(next_x)
        critical_scenarios.append(next_x)
        all_scores.append(score)

        if score > collision_threshold:
            cumulative_raw += 1
            fdc_curve_raw.append((evaluator.evaluation_count, cumulative_raw))
            if cumulative_raw >= 50 and n_50_raw == -1:
                n_50_raw = evaluator.evaluation_count

        # 每 batch_size 次迭代重新训练模型
        if (i - n_init + 1) % batch_size == 0:
            # 分层采样：保留所有危险点 + 最近的非危险点
            hazard_indices = [j for j, s in enumerate(all_scores) if s > collision_threshold]
            safe_indices = [j for j, s in enumerate(all_scores) if s <= collision_threshold]

            # 保留最近 300 个安全点 + 所有危险点
            if len(safe_indices) > 300:
                selected_safe = safe_indices[-300:]
            else:
                selected_safe = safe_indices

            selected_indices = sorted(hazard_indices + selected_safe)

            X_train = torch.tensor(np.array([critical_scenarios[j] for j in selected_indices]), dtype=torch.float32).to(device)
            Y_train = torch.tensor(np.array([all_scores[j] for j in selected_indices]), dtype=torch.float32).to(device)

            model.set_train_data(inputs=X_train, targets=Y_train, strict=False)

            # 快速训练
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            mll = ExactMarginalLogLikelihood(likelihood, model)

            for _ in range(20):
                optimizer.zero_grad()
                output = model(X_train)
                loss = -mll(output, Y_train)
                loss.backward()
                optimizer.step()

        if evaluator.real_simulation_count >= real_sim_budget:
            print(f"[!] Real simulation budget reached ({real_sim_budget}), stopping BO at iteration {i+1}/{n_calls}")
            break

    results = sorted(zip(critical_scenarios, all_scores), key=lambda x: x[1], reverse=True)
    sorted_scenarios = [x[0] for x in results]
    sorted_scores = [x[1] for x in results]

    return np.array(sorted_scenarios), np.array(sorted_scores), fdc_curve_raw, n_50_raw


if __name__ == '__main__':
    # 确保 matplotlib 在没有 GUI 的服务器上也能运行
    plt.switch_backend('agg')

    # 设置随机种子
    np.random.seed(42); random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='bo_search')
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
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--frame_skip', '-fs', type=int, default=1, help='skip of frame in each step')
    parser.add_argument('--port', type=int, default=2008)
    parser.add_argument('--tm_port', type=int, default=8008)
    parser.add_argument('--fixed_delta_seconds', type=float, default=0.1)
    parser.add_argument('--budget', type=int, default=2000, help='Total number of evaluations for BO')
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

    # Load configs
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

    # 运行贝叶斯优化
    critical_scenarios, all_scores, fdc_curve_raw, n_50_raw = run_bayesian_optimization(
        evaluator=evaluator,
        bounds=bounds,
        n_calls=args.budget,
        random_state=args.seed,
        output_dir=args.output_dir,
        real_sim_budget=args.real_sim_budget
    )

    end_time = time.time()
    search_time = end_time - start_time
    print(f"BO Execution time: {search_time:.2f} seconds")

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
        method_name='Bayesian Optimization',
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
    results_save_path = osp.join(args.output_dir, 'bayesian_optimization_results.pkl')
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