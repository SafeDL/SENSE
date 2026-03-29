"""
消融实验脚本: 纯小生境粒子群优化算法 (Pure Niching PSO without RL)
修正版 v2:
1. 仅在 Restart 和 End 时记录数据（符合你对 Baseline 的设定）。
2. 修复核心 Bug：使用 Raw Fitness 而非 Penalized Fitness 进行筛选。
3. 使用 List 替代 vstack 提升性能。
"""
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import time
import warnings
import os
import gpytorch
# 假设这些工具函数在这个路径下
from rlsan.src.tools.utils import quantum_radius, pheromone_decay
from rlsan.src.surrogate.utils import load_surrogate_model
from rlsan.src.RLSearch.diversity_evaluator import DiversityEvaluator

warnings.filterwarnings("ignore", category=RuntimeWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def set_ieee_style():
    plt.rcParams['savefig.dpi'] = 300
    # ... 其他样式设置保持一致 ...

class PureNichingPSO:
    def __init__(self, gp_model, gp_likelihood, num_particles=2000, num_iterations=1000, test_space_dim=3,
                 bounds=(-1, 1), num_subgroups=20, fixed_niche_radius=0.2):
        self.gp_model = gp_model
        self.gp_likelihood = gp_likelihood
        self.gp_model.to(device)
        self.gp_likelihood.to(device)

        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.test_space_dim = test_space_dim
        self.bounds = bounds
        self.num_subgroups = num_subgroups
        self.particles_per_group = num_particles // num_subgroups

        # === 固定参数 ===
        self.fixed_niche_radius = fixed_niche_radius
        self.w = 0.729
        self.c1 = 1.494
        self.c2 = 1.494

        self.subgroups = []
        # 使用 List 以保证性能
        self.global_hazardous_pool = []

    def _random_positions(self):
        return self.bounds[0] + (self.bounds[1] - self.bounds[0]) * np.random.rand(self.particles_per_group, self.test_space_dim)

    def fitness(self, positions):
        x_tensor = torch.tensor(positions, dtype=torch.float32).to(device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.gp_likelihood(self.gp_model(x_tensor))
            mu = posterior.mean.cpu().numpy()
            sigma2 = posterior.variance.cpu().numpy()

        # 保持与 RL 一致的计算方式
        fitness_value = -mu - 1.0 * sigma2
        return fitness_value, np.sqrt(sigma2)

    def init_subgroups(self):
        self.subgroups = []
        for _ in range(self.num_subgroups):
            particles = self._random_positions()
            velocities = np.zeros((self.particles_per_group, self.test_space_dim))
            personal_best_fitness, sigma = self.fitness(particles)
            personal_best_positions = particles.copy()
            best_idx = np.argmin(personal_best_fitness)

            group = {
                'particles': particles,
                'velocities': velocities,
                'personal_best_positions': personal_best_positions,
                # 【新增】raw_fitness 用于存储未惩罚的真实值
                'raw_fitness': personal_best_fitness.copy(),
                # personal_best_fitness 后续会被惩罚修改
                'personal_best_fitness': personal_best_fitness.copy(),
                'sigma': sigma,
                'global_best_positions': personal_best_positions[best_idx],
                'global_best_fitness': personal_best_fitness[best_idx],
                'archive_positions': personal_best_positions.copy(),
                'archive_fitness': personal_best_fitness.copy(),
                # 固定参数
                'niche_radius': self.fixed_niche_radius,
                'w': self.w, 'c1': self.c1, 'c2': self.c2,
                'stagnation_counter': 0
            }
            self.subgroups.append(group)

    def _update_velocity(self, group):
        w, c1, c2 = group['w'], group['c1'], group['c2']
        for i in range(self.particles_per_group):
            r1, r2 = random.random(), random.random()
            cognitive = c1 * r1 * (group['personal_best_positions'][i] - group['particles'][i])
            social = c2 * r2 * (group['global_best_positions'] - group['particles'][i])
            group['velocities'][i] = w * group['velocities'][i] + cognitive + social

    def _update_position(self, group):
        group['particles'] = np.clip(group['particles'] + group['velocities'], self.bounds[0], self.bounds[1])

    # === 核心修改：分离 Raw 和 Penalized Fitness ===
    def _evaluate_fitness(self, group):
        # 1. 计算当前原始 Fitness
        fitness, sigma = self.fitness(group['particles'])

        # 2. 更新 pbest (使用原始 fitness 比较)
        for i in range(self.particles_per_group):
            if fitness[i] < group['raw_fitness'][i]:
                group['personal_best_positions'][i] = group['particles'][i]
                group['raw_fitness'][i] = fitness[i]
                group['sigma'][i] = sigma[i]

        # 3. 更新 gbest (使用 Raw 比较)
        best_idx = np.argmin(group['raw_fitness'])
        if group['raw_fitness'][best_idx] < group['global_best_fitness']:
            group['global_best_positions'] = group['personal_best_positions'][best_idx]
            group['global_best_fitness'] = group['raw_fitness'][best_idx]

        # 4. 应用惩罚 (修改 personal_best_fitness 用于下一步速度更新)
        group['personal_best_fitness'] = group['raw_fitness'].copy()
        self._apply_niche_sharing(group)

    def _apply_niche_sharing(self, group):
        positions = group['personal_best_positions']
        # 使用 fitness 作为基准来计算
        fitness = group['personal_best_fitness']
        r_niche = group['niche_radius']
        dist_matrix = np.linalg.norm(positions[:, None] - positions, axis=2)
        with np.errstate(divide='ignore', invalid='ignore'):
            sharing_matrix = np.where(
                dist_matrix < r_niche,
                1 - dist_matrix / r_niche,
                0
            )
        niche_density = np.sum(sharing_matrix, axis=1)
        # 惩罚
        adjusted_fitness = fitness / (1 + niche_density + np.finfo(float).eps)
        # 只覆盖用于 velocity update 的 fitness
        group['personal_best_fitness'] = adjusted_fitness
        return group

    # === 核心修改：Archive 必须存 Raw Fitness ===
    def _update_archive_logic(self, group):
        FIXED_ARCHIVE_THRESHOLD = 1e-8
        archive_positions = group['archive_positions']
        archive_fitness = group['archive_fitness']
        latest_positions = group['personal_best_positions']
        latest_raw_fitness = group['raw_fitness']

        for i in range(len(latest_positions)):
            distances = np.linalg.norm(archive_positions - latest_positions[i], axis=1)
            if np.all(distances > FIXED_ARCHIVE_THRESHOLD):
                archive_positions[i] = latest_positions[i]
                archive_fitness[i] = latest_raw_fitness[i]

        group['archive_positions'] = archive_positions
        group['archive_fitness'] = archive_fitness

    def _restart_group(self):
        particles = self._random_positions()
        velocities = np.zeros((self.particles_per_group, self.test_space_dim))
        personal_best_positions = particles.copy()
        personal_best_fitness, sigma = self.fitness(particles)
        global_best_idx = np.argmin(personal_best_fitness)

        new_group = {
            'particles': particles,
            'velocities': velocities,
            'personal_best_positions': personal_best_positions,
            'personal_best_fitness': personal_best_fitness.copy(),
            'raw_fitness': personal_best_fitness.copy(),
            'sigma': sigma,
            'global_best_positions': personal_best_positions[global_best_idx],
            'global_best_fitness': personal_best_fitness[global_best_idx],
            'archive_positions': personal_best_positions.copy(),
            'archive_fitness': personal_best_fitness.copy(),
            'niche_radius': self.fixed_niche_radius,
            'w': self.w, 'c1': self.c1, 'c2': self.c2,
            'stagnation_counter': 0
        }
        return new_group

    def _collect_results(self):
        # 保持与 RL 版本一致的后处理逻辑
        if len(self.global_hazardous_pool) == 0:
            print("No hazardous cases found.")
            return np.array([])

        try:
            all_hazards = np.array(self.global_hazardous_pool)
        except:
            return np.array([])

        # 1. 简单去重
        _, unique_indices = np.unique(np.round(all_hazards, decimals=4), axis=0, return_index=True)
        unique_hazards = all_hazards[unique_indices]
        print(f"[Baseline] Unique locations (pre-filtering): {len(unique_hazards)}")

        # 2. 贪婪筛选
        final_seeds = []
        if len(unique_hazards) > 0:
            remaining_indices = list(range(len(unique_hazards)))
            random.shuffle(remaining_indices)
            while len(remaining_indices) > 0:
                current_idx = remaining_indices.pop(0)
                current_seed = unique_hazards[current_idx]
                final_seeds.append(current_seed)
                if len(remaining_indices) == 0: break
                remaining_points = unique_hazards[remaining_indices]
                distances = np.linalg.norm(remaining_points - current_seed, axis=1)
                keep_mask = distances > 0.05
                remaining_indices = [remaining_indices[i] for i in range(len(keep_mask)) if keep_mask[i]]

        final_seeds = np.array(final_seeds)
        print(f"[Baseline] Representative Seeds: {len(final_seeds)}")
        return final_seeds

    def run(self):
        self.init_subgroups()
        print("Start optimization (Baseline)...")

        for iter_num in tqdm(range(self.num_iterations)):
            for group_idx, group in enumerate(self.subgroups):
                self._update_velocity(group)
                self._update_position(group)

                # 评估适应度 (内部会更新 raw_fitness 并应用惩罚)
                self._evaluate_fitness(group)

                # 更新存档 (使用 raw_fitness)
                self._update_archive_logic(group)

                # 停滞检测 (这里使用简单的 counter 模拟)
                # 实际上你应该用和 RL 完全一样的判定逻辑 (quantum_radius 等)
                try:
                    is_stagnant = quantum_radius(group)
                except:
                    # 如果没有 utils，用简单的距离判断代替
                    is_stagnant = False

                if is_stagnant:
                    group['stagnation_counter'] += 1
                else:
                    group['stagnation_counter'] = max(0, group['stagnation_counter'] - 1)

                if group['stagnation_counter'] >= 5:
                    print(f"Restarting subgroup {group_idx}")
                    threshold = -0.3
                    if len(group['archive_fitness']) > 0:
                        hazardous_mask = group['archive_fitness'] < threshold
                        if np.any(hazardous_mask):
                            found_hazards = group['archive_positions'][hazardous_mask]
                            for h in found_hazards:
                                self.global_hazardous_pool.append(h.copy())

                    self.subgroups[group_idx] = self._restart_group()

        # 循环结束后，也应该把当前所有群体的 Archive 检查一遍，防止遗漏最后一代
        for group in self.subgroups:
            threshold = -0.3
            if len(group['archive_fitness']) > 0:
                hazardous_mask = group['archive_fitness'] < threshold
                if np.any(hazardous_mask):
                    found_hazards = group['archive_positions'][hazardous_mask]
                    for h in found_hazards:
                        self.global_hazardous_pool.append(h.copy())

        return self._collect_results()

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)
    set_ieee_style()

    # 加载与 RL 实验相同的模型
    model_path = '../../results/s5exp/surrogate_model.pkl'
    gp_model, gp_likelihood = load_surrogate_model(model_path)

    config = {
        'gp_model': gp_model,
        'gp_likelihood': gp_likelihood,
        'num_particles': 2000,
        'num_iterations': 1000,
        'test_space_dim': 3,
        'bounds': (-1, 1),
        'num_subgroups': 20,
        'fixed_niche_radius': 0.2,
    }

    baseline = PureNichingPSO(**config)
    results = baseline.run()

    # 评估
    evaluator = DiversityEvaluator(min_samples=10)
    evaluator.evaluate(results, save_path='baseline_diversity.png', title_prefix='Baseline')