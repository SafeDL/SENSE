"""
Learning to Collide (LC) 基线方法

基于 REINFORCE 策略梯度的对抗性初始状态生成方法。
使用训练好的 LC 模型生成对抗性初始位置参数，通过 CARLA 仿真或代理模型评估危险性。

参考论文:
    Learning to Collide: An Adaptive Safety-Critical Scenarios Generating Method
    <https://arxiv.org/pdf/2003.01197.pdf>
"""

import pickle
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
from safebench.scenario.scenario_policy.reinforce_continuous import REINFORCE, normalize_routes
from safebench.util.torch_util import CUDA, CPU
from safebench.gym_carla.env_wrapper import VectorWrapper
from safebench.util.logger import Logger, setup_logger_kwargs
from safebench.scenario.tools.scenario_utils import scenario_parse_simple
from surrogate_utils import SurrogateModel, SurrogateEvaluator, prepare_baseline_results
import carla


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


class LCScenarioGenerator:
    """
    使用训练好的 LC (REINFORCE) 模型生成对抗性初始状态参数。
    将 LC 模型的输出（初始状态动作）转换为与其他 baseline 对齐的 3D/4D 参数空间。
    """

    def __init__(self, model_path, model_id='weight', lr=0.004,
                 standard_action_dim=True, num_scenario=1):
        """
        Args:
            model_path: LC 模型权重所在目录（绝对路径）
            model_id: 模型文件名前缀
            lr: 学习率（加载优化器用）
            standard_action_dim: True=4维动作, False=3维动作
            num_scenario: 场景数量（固定为1与其他baseline对齐）
        """
        self.standard_action_dim = standard_action_dim
        self.num_scenario = num_scenario
        self.action_dim = 4 if standard_action_dim else 3

        # 构造与 REINFORCE 兼容的 scenario_config
        scenario_config = {
            'num_scenario': num_scenario,
            'batch_size': 16,
            'model_path': model_path,  # 将在 REINFORCE.__init__ 中与ROOT_DIR拼接
            'model_id': model_id,
            'lr': lr,
            'standard_action_dim': standard_action_dim,
            'ROOT_DIR': '',  # model_path 已经是绝对路径，ROOT_DIR 置空
        }

        # 使用一个简单的logger占位
        class SimpleLogger:
            def log(self, msg, color=None):
                print(msg)

        self.logger = SimpleLogger()
        self.policy = REINFORCE(scenario_config, self.logger)

        # 加载模型权重
        model_filename = osp.join(model_path, f'{model_id}.pt')
        if osp.exists(model_filename):
            print(f'[✓] Loading LC model from {model_filename}')
            with open(model_filename, 'rb') as f:
                checkpoint = torch.load(f, map_location='cpu')
            self.policy.model = CUDA(self.policy.model)
            self.policy.model.load_state_dict(checkpoint['parameters'])
            if 'value_parameters' in checkpoint:
                self.policy.value_net.load_state_dict(checkpoint['value_parameters'])
            self.policy.model.eval()
            print(f'[✓] LC model loaded successfully (action_dim={self.action_dim})')
        else:
            print(f'[!] LC model not found at {model_filename}, using random initialization')

    def generate_actions(self, n_samples, static_obs=None, deterministic=False):
        """
        使用 LC 模型批量生成对抗性初始状态动作。

        Args:
            n_samples: 生成的样本数量
            static_obs: 静态观测列表，每个元素是 {'route': array, 'target_speed': float}
            deterministic: 是否使用确定性策略

        Returns:
            actions: shape (n_samples, action_dim), 值在 [-1, 1] 范围内
        """
        if static_obs is None:
            raise ValueError("static_obs (route state) is required for LC model")

        route_states = []
        for obs in static_obs:
            if isinstance(obs, dict):
                route = obs['route']
                target_speed = obs.get('target_speed', 25.0)
            else:
                route = obs
                target_speed = 25.0

            # 等分路线取 num_waypoint 个点（与 REINFORCE 一致）
            index = np.linspace(1, len(route) - 1, self.policy.num_waypoint).tolist()
            index = [int(i) for i in index]
            route_sampled = route[index]

            # 归一化路线（与 REINFORCE 一致）
            route_norm = normalize_routes(route_sampled)[:, 0]  # shape: (num_waypoint*2,)

            # 归一化速度（与 REINFORCE 一致）
            target_speed_norm = target_speed / 36.0

            # 拼接状态
            state = np.concatenate((route_norm, [target_speed_norm]), axis=0).astype('float32')
            route_states.append(state)

        static_obs_array = np.array(route_states, dtype='float32')
        processed_state = CUDA(torch.from_numpy(static_obs_array))

        with torch.no_grad():
            _, _, actions = self.policy.model.forward(processed_state, deterministic)

        actions_np = CPU(actions)
        actions_np = np.clip(actions_np, -1.0, 1.0)

        return actions_np


def run_lc_search(evaluator, lc_generator, scenario_config, bounds, n_calls=1000,
                  output_dir='log', real_sim_budget=2000, niche_radius=0.05,
                  collision_threshold=0.3, checkpoint_interval=50):
    """
    使用 LC (REINFORCE) 模型生成对抗性场景并评估。

    直接使用 runner 中的 scenario_policy 和 env，避免重复创建 CARLA 客户端。

    Args:
        evaluator: SurrogateEvaluator 实例
        runner: CarlaRunner 实例（包含 scenario_policy 和 env）
        scenario_config: 场景配置对象
        bounds: 搜索空间界限
        n_calls: 总评估次数预算
        output_dir: 输出目录
        real_sim_budget: 真值仿真调用上限
        niche_radius: 小生境半径
        collision_threshold: 碰撞阈值
        checkpoint_interval: 检查点保存间隔

    Returns:
        critical_scenarios, all_scores, fdc_info
    """
    checkpoint_path = osp.join(output_dir, 'lc_checkpoint.pkl')

    # 初始化变量
    start_iter = 0
    critical_scenarios = []
    all_scores = []
    fdc_curve_raw = []
    fdc_curve_representative = []
    n_50_raw = -1
    n_50_representative = -1
    hazardous_set = []
    hazardous_scores_list = []
    cumulative_raw = 0
    last_rep_count = 0

    # 尝试加载检查点
    if osp.exists(checkpoint_path):
        try:
            with open(checkpoint_path, 'rb') as f:
                ckpt = pickle.load(f)
            start_iter = ckpt['iteration']
            critical_scenarios = ckpt['critical_scenarios']
            all_scores = ckpt['all_scores']
            fdc_curve_raw = ckpt['fdc_curve_raw']
            fdc_curve_representative = ckpt['fdc_curve_representative']
            n_50_raw = ckpt['n_50_raw']
            n_50_representative = ckpt['n_50_representative']
            hazardous_set = ckpt['hazardous_set']
            hazardous_scores_list = ckpt['hazardous_scores_list']
            cumulative_raw = ckpt['cumulative_raw']
            last_rep_count = ckpt['last_rep_count']
            evaluator.evaluation_count = ckpt['eval_count']
            evaluator.surrogate_call_count = ckpt['surrogate_call_count']
            evaluator.real_simulation_count = ckpt['real_sim_count']
            print(f"[✓] Resumed from checkpoint: iteration {start_iter}/{n_calls}")
            print(f"    Evaluations: {evaluator.evaluation_count}, Real sims: {evaluator.real_simulation_count}, Hazardous: {cumulative_raw}")
        except Exception as e:
            print(f"[!] Failed to load checkpoint: {e}, starting fresh")
            start_iter = 0

    print(f"Starting Learning to Collide Search with budget: {n_calls}")
    print(f"Real simulation budget limit: {real_sim_budget}")
    print("-" * 60)

    # 确保环境已初始化（CarlaRunner_simple 在 __init__ 中初始化了）
    if evaluator.real_runner.env is None:
        from safebench.gym_carla.env_wrapper import VectorWrapper
        evaluator.real_runner.env = VectorWrapper(
            evaluator.real_runner.env_params,
            evaluator.real_runner.scenario_config,
            evaluator.real_runner.world,
            None,
            evaluator.real_runner.display,
            evaluator.real_runner.logger
        )

    for i in tqdm(range(start_iter, n_calls), desc="LC Search", unit="eval", initial=start_iter, total=n_calls):
        # 获取 static_obs（环境由 evaluator 管理）
        sampled_scenario_configs = [scenario_config]
        static_obs = evaluator.real_runner.env.get_static_obs(sampled_scenario_configs)

        # 使用 LC 模型生成初始状态动作
        scenario_init_action = lc_generator.generate_actions(
            1, static_obs=static_obs, deterministic=False
        )

        # 提取动作（截取到搜索空间维度）
        action = scenario_init_action[0][:len(bounds)]
        next_x = np.clip(action, bounds[0][0], bounds[0][1])

        # 评估
        current_score = evaluator.evaluate(next_x)

        critical_scenarios.append(next_x.copy())
        all_scores.append(current_score)

        # FDC 统计
        if current_score > collision_threshold:
            cumulative_raw += 1
            hazardous_set.append(next_x.copy())
            hazardous_scores_list.append(current_score)

        search_budget = evaluator.evaluation_count

        fdc_curve_raw.append((search_budget, cumulative_raw))
        if cumulative_raw >= 50 and n_50_raw == -1:
            n_50_raw = search_budget

        # 每 50 次更新代表性点
        if len(hazardous_set) > 0 and (i + 1) % 50 == 0:
            hp_arr = np.array(hazardous_set)
            hs_arr = np.array(hazardous_scores_list)
            rep_info = select_representative_seeds(hp_arr, -hs_arr, niche_radius=niche_radius)
            last_rep_count = len(rep_info['representative'])

        fdc_curve_representative.append((search_budget, last_rep_count))
        if last_rep_count >= 50 and n_50_representative == -1:
            n_50_representative = search_budget

        # 保存检查点
        if (i + 1) % checkpoint_interval == 0:
            ckpt = {
                'iteration': i + 1,
                'critical_scenarios': critical_scenarios,
                'all_scores': all_scores,
                'fdc_curve_raw': fdc_curve_raw,
                'fdc_curve_representative': fdc_curve_representative,
                'n_50_raw': n_50_raw,
                'n_50_representative': n_50_representative,
                'hazardous_set': hazardous_set,
                'hazardous_scores_list': hazardous_scores_list,
                'cumulative_raw': cumulative_raw,
                'last_rep_count': last_rep_count,
                'eval_count': evaluator.evaluation_count,
                'surrogate_call_count': evaluator.surrogate_call_count,
                'real_sim_count': evaluator.real_simulation_count,
            }
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(ckpt, f)

        # 真值仿真熔断
        if evaluator.real_simulation_count >= real_sim_budget:
            print(f"[!] Real simulation budget reached ({real_sim_budget}), stopping at iteration {i+1}/{n_calls}")
            break

    # 排序返回
    results = sorted(zip(critical_scenarios, all_scores), key=lambda x: x[1], reverse=True)
    sorted_scenarios = [x[0] for x in results]
    sorted_scores = [x[1] for x in results]

    fdc_info = {
        'fdc_curve_raw': fdc_curve_raw,
        'fdc_curve_representative': fdc_curve_representative,
        'n_50_raw': n_50_raw,
        'n_50_representative': n_50_representative,
    }

    return np.array(sorted_scenarios), np.array(sorted_scores), fdc_info


if __name__ == '__main__':
    plt.switch_backend('agg')
    np.random.seed(42); random.seed(42)

    parser = argparse.ArgumentParser(description='Learning to Collide Baseline')
    parser.add_argument('--exp_name', type=str, default='lc_search')
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
    parser.add_argument('--render', type=bool, default=True)
    parser.add_argument('--frame_skip', '-fs', type=int, default=1, help='skip of frame in each step')
    parser.add_argument('--port', type=int, default=2008)
    parser.add_argument('--tm_port', type=int, default=8008)
    parser.add_argument('--fixed_delta_seconds', type=float, default=0.1)
    parser.add_argument('--budget', type=int, default=1000000, help='Total number of evaluations')
    parser.add_argument('--surrogate_model_path', type=str,
                       default='rlsan/results/s1exp/surrogate_model_1000.pkl',
                       help='Path to surrogate model')
    parser.add_argument('--real_sim_budget', type=int, default=2000,
                        help='Max real CARLA simulations before stopping')
    parser.add_argument('--uncertainty_threshold', type=float, default=0.00217,
                        help='Uncertainty threshold for triggering real simulation (variance)')
    # LC 特有参数
    parser.add_argument('--lc_model_dir', type=str, default='safebench/scenario/scenario_data/model_ckpt/lc',
                        help='LC model checkpoint directory')
    parser.add_argument('--lc_model_id', type=str, default='scenario_01_weight',
                        help='LC model file name (without .pt extension)')
    parser.add_argument('--lc_standard_action_dim', type=bool, default=False,
                        help='True=4D action, False=3D action')

    args = parser.parse_args()
    args_dict = vars(args)

    # 设置输出目录
    args.output_dir = osp.join(args.ROOT_DIR, 'log', 'baselines')
    os.makedirs(args.output_dir, exist_ok=True)

    set_torch_variable(args.device)
    torch.set_num_threads(args.threads)
    set_seed(args.seed)

    # 加载配置
    agent_config_path = osp.join(args.ROOT_DIR, 'safebench/agent/config', args.agent_cfg)
    agent_config = load_config(agent_config_path)
    agent_config.update(args_dict)

    scenario_config_path = osp.join(args.ROOT_DIR, 'rlsan/config', args.scenario_cfg)
    scenario_config = load_config(scenario_config_path)
    scenario_config.update(args_dict)

    # 使用 scenario_parse_simple 加载正确的场景配置对象（包含 trajectory）
    class SimpleLogger:
        @staticmethod
        def log(msg, *args, **kwargs):
            del args, kwargs
            print(msg)

    logger = SimpleLogger()
    config_by_map = scenario_parse_simple(scenario_config, logger)

    # 获取第一个地图的第一个场景配置
    map_name = list(config_by_map.keys())[0]
    parsed_scenario_config = config_by_map[map_name][0]

    runner = CarlaRunner(agent_config, scenario_config, step_by_step=True)
    bounds = [(-1, 1), (-1, 1), (-1, 1)]

    # 初始化 LC 场景生成器
    lc_model_path = osp.join(args.ROOT_DIR, args.lc_model_dir)
    lc_generator = LCScenarioGenerator(
        model_path=lc_model_path,
        model_id=args.lc_model_id,
        standard_action_dim=args.lc_standard_action_dim,
        num_scenario=args.num_scenario,
    )

    # 初始化评估器
    evaluator = None
    try:
        surrogate_path = osp.abspath(osp.join(args.ROOT_DIR, args.surrogate_model_path))
        if osp.exists(surrogate_path):
            print(f"[*] Loading surrogate model from: {surrogate_path}")
            surrogate = SurrogateModel(surrogate_path)
            evaluator = SurrogateEvaluator(surrogate, real_runner=runner, use_real_sim=True,
                                          uncertainty_threshold=args.uncertainty_threshold)
            print(f"[✓] Surrogate evaluator initialized")
    except Exception as e:
        print(f"  (Error loading surrogate: {e})")

    if evaluator is None:
        raise RuntimeError("Failed to initialize evaluator")

    start_time = time.time()
    critical_scenarios, all_scores, fdc_info = run_lc_search(
        evaluator=evaluator,
        lc_generator=lc_generator,
        scenario_config=parsed_scenario_config,
        bounds=bounds,
        n_calls=args.budget,
        output_dir=args.output_dir,
        real_sim_budget=args.real_sim_budget,
        niche_radius=0.05
    )

    end_time = time.time()
    search_time = end_time - start_time
    print(f"LC Search Execution time: {search_time:.2f} seconds")

    # 获取评估统计信息
    eval_stats = evaluator.get_stats() if evaluator else {
        'total_evaluations': args.budget,
        'surrogate_calls': 0,
        'real_simulations': args.budget
    }

    # 提取危险样本
    collision_mask = all_scores > 0.3
    hazardous_scenarios = critical_scenarios[collision_mask]
    hazardous_scores = all_scores[collision_mask]

    # 去重处理
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
        method_name='Learning to Collide',
        search_time=search_time,
        grid_x_data=None,
        grid_y_data=None,
        all_scores_history=np.array(all_scores)
    )

    # 添加代表性失效信息
    baseline_results['representative_failures_count'] = len(representative_scenarios)

    # 覆盖动态FDC数据
    baseline_results['fdc_curve_raw'] = fdc_info['fdc_curve_raw']
    baseline_results['fdc_curve_representative'] = fdc_info['fdc_curve_representative']
    baseline_results['n_50_raw'] = fdc_info['n_50_raw']
    baseline_results['n_50_representative'] = fdc_info['n_50_representative']

    # 计算 AUC
    if len(fdc_info['fdc_curve_raw']) > 1:
        budgets = np.array([p[0] for p in fdc_info['fdc_curve_raw']])
        failures = np.array([p[1] for p in fdc_info['fdc_curve_raw']])
        baseline_results['auc_fdc_raw'] = float(np.trapz(failures, budgets))
    else:
        baseline_results['auc_fdc_raw'] = 0.0

    if len(fdc_info['fdc_curve_representative']) > 1:
        budgets = np.array([p[0] for p in fdc_info['fdc_curve_representative']])
        failures = np.array([p[1] for p in fdc_info['fdc_curve_representative']])
        baseline_results['auc_fdc_representative'] = float(np.trapz(failures, budgets))
    else:
        baseline_results['auc_fdc_representative'] = 0.0

    # 保存结果
    results_save_path = osp.join(args.output_dir, 'learning_to_collide_results.pkl')
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
    if fdc_info['n_50_raw'] == -1:
        print(f"N_50 (Raw): Not reached (found {baseline_results['raw_failures_count']}/50)")
    else:
        print(f"N_50 (Raw): {fdc_info['n_50_raw']}")

    print(f"\n[FDC Metrics - Representative Failures]")
    print(f"AUC-FDC (Representative): {baseline_results['auc_fdc_representative']:.4e}")
    if fdc_info['n_50_representative'] == -1:
        print(f"N_50 (Representative): Not reached (found {baseline_results['representative_failures_count']}/50)")
    else:
        print(f"N_50 (Representative): {fdc_info['n_50_representative']}")
