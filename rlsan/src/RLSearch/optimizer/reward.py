"""
reward.py - 统一安全搜索奖励计算器

核心设计：
1. 统一尺度：将奖励限制在 [-1, 10] 区间，确保不同算法下的 Q 值/Advantage 具有可比性。
2. 核心指标：
   - 危险小生境发现 (Danger Niche)：最高优先级的稀疏奖励。
   - 危险区域覆盖 (Danger Coverage)：中等优先级的密集奖励。
   - 进化状态匹配 (State Match)：辅助引导奖励，帮助 Agent 学会控制 PSO 参数。
   - 适应度改进 (Fitness Improvement)：基础奖励，使用 Log 尺度防止后期奖励消失。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
from .evo_state import EvolutionaryState, StateInfo


class ProjectionCoverageTracker:
    """
    投影覆盖率追踪器（高维友好）
    
    使用多个2D投影来估算高维空间覆盖率，避免指数爆炸：
    - 对于d维空间，选择 d*(d-1)/2 个2D投影
    - 每个2D投影使用固定分辨率网格
    - 总覆盖率 = 各投影覆盖率的平均
    """
    
    def __init__(self,
                 bounds: Tuple[float, float],
                 dim: int,
                 resolution: int = 20):
        """
        Args:
            bounds: 搜索空间边界
            dim: 维度
            resolution: 2D投影的分辨率
        """
        self.bounds = bounds
        self.dim = dim
        self.resolution = resolution
        self.range = bounds[1] - bounds[0]
        
        # 选择投影对（最多20对以控制开销）
        self.projection_pairs = []
        for i in range(min(dim, 6)):
            for j in range(i + 1, min(dim, 6)):
                self.projection_pairs.append((i, j))
        if not self.projection_pairs:
            self.projection_pairs = [(0, min(1, dim-1))]
        
        # 每个投影的已访问网格
        self.visited_grids = [set() for _ in self.projection_pairs]
        self.total_cells_per_projection = resolution * resolution
        
        # 总访问位置计数（用于平滑统计）
        self.total_positions = 0
        self.unique_hashes = set()  # 简单哈希去重
    
    def update(self, positions: np.ndarray) -> float:
        """
        更新覆盖率
        
        Args:
            positions: (n, dim) 位置数组
            
        Returns:
            increment: 覆盖率增量
        """
        old_coverages = [len(g) for g in self.visited_grids]
        
        # 归一化位置
        normalized = (positions - self.bounds[0]) / self.range
        normalized = np.clip(normalized, 0, 0.999)  # 避免边界问题
        
        # 对每个投影更新
        for proj_idx, (i, j) in enumerate(self.projection_pairs):
            # 获取这两个维度
            proj_coords = normalized[:, [i, j]]
            # 转换为网格索引
            grid_indices = (proj_coords * self.resolution).astype(int)
            # 添加到集合
            for idx in grid_indices:
                self.visited_grids[proj_idx].add(tuple(idx))
        
        # 简单哈希位置用于统计
        for pos in positions:
            h = hash(tuple(np.round(pos, decimals=1)))
            self.unique_hashes.add(h)
        
        self.total_positions += len(positions)
        
        # 计算增量
        new_coverages = [len(g) for g in self.visited_grids]
        total_increment = sum(new_coverages) - sum(old_coverages)
        increment = total_increment / (len(self.projection_pairs) * self.total_cells_per_projection)
        
        return increment
    
    def get_coverage_rate(self) -> float:
        """获取当前平均覆盖率"""
        total_visited = sum(len(g) for g in self.visited_grids)
        total_cells = len(self.projection_pairs) * self.total_cells_per_projection
        return total_visited / total_cells
    
    def get_unique_region_count(self) -> int:
        """获取唯一区域数量"""
        return len(self.unique_hashes)
    
    def reset(self):
        """重置"""
        for g in self.visited_grids:
            g.clear()
        self.unique_hashes.clear()
        self.total_positions = 0


class DangerNicheTracker:
    """危险小生境追踪器 (复用原 PPO 版逻辑)"""
    MAX_STORED_POSITIONS = 500
    
    def __init__(self, niche_radius: float = 0.3, danger_threshold: float = -0.3):
        self.niche_radius = niche_radius
        self.danger_threshold = danger_threshold
        self.danger_niches: List[Tuple[np.ndarray, float]] = []
        self._cached_coverage = 0.0
        self.total_danger_points_found = 0
    
    def update(self, positions: np.ndarray, fitness_values: np.ndarray) -> Dict:
        danger_mask = fitness_values < self.danger_threshold
        danger_positions = positions[danger_mask]
        danger_fitness = fitness_values[danger_mask]
        
        if len(danger_positions) == 0:
            return {'new_niches': 0, 'coverage_increment': 0.0}
        
        self.total_danger_points_found += len(danger_positions)
        new_niches = 0
        
        if self.danger_niches:
            niche_centers = np.array([n[0] for n in self.danger_niches])
        else:
            niche_centers = None
            
        for i, pos in enumerate(danger_positions):
            fit = danger_fitness[i]
            merged = False
            if niche_centers is not None:
                dists = np.linalg.norm(niche_centers - pos, axis=1)
                min_idx = np.argmin(dists)
                if dists[min_idx] < self.niche_radius:
                    if fit < self.danger_niches[min_idx][1]:
                        self.danger_niches[min_idx] = (pos.copy(), fit)
                        niche_centers[min_idx] = pos
                    merged = True
            
            if not merged:
                self.danger_niches.append((pos.copy(), fit))
                if niche_centers is not None:
                    niche_centers = np.vstack([niche_centers, pos])
                else:
                    niche_centers = pos.reshape(1, -1)
                new_niches += 1
                
        # 计算覆盖率增量
        prev_cov = self._cached_coverage
        self._cached_coverage = self._compute_coverage()
        inc = self._cached_coverage - prev_cov
        
        return {'new_niches': new_niches, 'coverage_increment': inc}

    def _compute_coverage(self) -> float:
        if len(self.danger_niches) < 2: return 0.0
        centers = np.array([n[0] for n in self.danger_niches])
        var = np.var(centers, axis=0).mean()
        return min(1.0, var / 0.5)

    def reset(self):
        self.danger_niches.clear()
        self._cached_coverage = 0.0
        self.total_danger_points_found = 0
        
    def get_num_niches(self) -> int:
        """Alias for get_num_danger_niches to be compatible with NicheDiscoveryTracker interface"""
        return len(self.danger_niches)

class SafetyRewardCalculator:
    """
    统一安全搜索奖励计算器
    
    奖励组成：
    1. r_danger_niche: 发现新危险小生境 (+5.0 * count)
    2. r_danger_cov: 危险区域覆盖率增加 (+10.0 * inc)
    3. r_fitness: 适应度改进 (log scale)
    4. r_state_match: 进化状态参数匹配 (max +0.5)
    """
    
    DEFAULT_WEIGHTS = {
        'danger_niche': 5.0,
        'danger_cov': 5.0,
        'fitness': 1.0,
        'state_match': 0.5,
        'milestone': 2.0
    }
    
    def __init__(self, 
                 bounds: Tuple[float, float], 
                 dim: int,
                 niche_radius: float = 0.1,
                 danger_threshold: float = -0.3,
                 weights: Dict[str, float] = None):
        self.bounds = bounds
        self.dim = dim
        self.danger_threshold = danger_threshold
        self.weights = self.DEFAULT_WEIGHTS.copy()
        if weights: self.weights.update(weights)
        
        # 追踪器
        self.danger_tracker = DangerNicheTracker(niche_radius, danger_threshold)
        self.coverage_tracker = ProjectionCoverageTracker(bounds, dim)
        
        # [Compatibility] Alias danger_tracker as niche_tracker so StateAwareNichePSO can use it
        self.niche_tracker = self.danger_tracker
        
        # 统计
        self.stats = {k: deque(maxlen=100) for k in self.weights.keys()}
        self.stats['total'] = deque(maxlen=100)
        self.milestones = {'first_danger': False, 'danger_5': False, 'high_cov': False}
        self.best_fitness = float('inf')

    def compute(self,
                prev_best: float,
                current_best: float,
                action_idx,
                state_info: Optional[StateInfo] = None,
                positions: np.ndarray = None,
                fitness_values: np.ndarray = None,
                **kwargs) -> float:
        
        components = {}
        
        # 1. 危险小生境奖励 (High Priority)
        r_niche = 0.0
        r_cov = 0.0
        if positions is not None and fitness_values is not None:
            info = self.danger_tracker.update(positions, fitness_values)
            r_niche = info['new_niches'] * self.weights['danger_niche']
            r_cov = info['coverage_increment'] * 10.0 * self.weights['danger_cov']
            
        components['danger_niche'] = r_niche
        components['danger_cov'] = r_cov
        
        # 2. 适应度改进奖励 (Base)
        delta = prev_best - current_best
        r_fit = 0.0
        if delta > 1e-8:
            r_fit = np.log1p(delta * 100) * self.weights['fitness']
        elif current_best < self.danger_threshold:
            # 在危险区域内保持也给少量分
            r_fit = 0.1 * self.weights['fitness']
        components['fitness'] = r_fit
        
        # 3. 状态匹配奖励 (Guidance)
        r_match = 0.0
        if state_info:
            r_match = self._compute_state_match(action_idx, state_info)
        components['state_match'] = r_match
        
        # 4. 里程碑奖励 (One-time)
        r_milestone = 0.0
        num_niches = len(self.danger_tracker.danger_niches)
        if num_niches >= 1 and not self.milestones['first_danger']:
            r_milestone += self.weights['milestone']
            self.milestones['first_danger'] = True
        if num_niches >= 5 and not self.milestones['danger_5']:
            r_milestone += self.weights['milestone'] * 2
            self.milestones['danger_5'] = True
        components['milestone'] = r_milestone
        
        # 汇总
        total = sum(components.values())
        
        # 记录统计
        for k, v in components.items():
            if k in self.stats: self.stats[k].append(v)
        self.stats['total'].append(total)
        
        if current_best < self.best_fitness:
            self.best_fitness = current_best
            
        return np.clip(total, -1.0, 10.0)

    def _compute_state_match(self, action, state_info: StateInfo) -> float:
        """计算动作与进化状态的匹配度"""
        # 支持连续动作(PPO)和离散动作(DDQN)
        if isinstance(action, (np.ndarray, list)):
            # Continuous: [w, c1, c2, vs]
            # 简化版高斯评分
            w = action[0]
            target_w = 0.7 if state_info.state == EvolutionaryState.EXPLORATION else 0.4
            score = np.exp(-(w - target_w)**2 / 0.1)
            return score * self.weights['state_match'] * state_info.confidence
        else:
            # Discrete: index
            # 简单的规则匹配 (假设 action 3=Exploration, 4=Exploitation...)
            # 具体映射需由于外部定义，这里给一个基础分
            return 0.1 * self.weights['state_match']

    def compute_batch(self, 
                      prev_bests: np.ndarray,
                      current_bests: np.ndarray,
                      actions,
                      state_infos: List[StateInfo] = None,
                      positions_list: List[np.ndarray] = None,
                      fitness_list: List[np.ndarray] = None,
                      **kwargs) -> np.ndarray:
        rewards = []
        for i in range(len(prev_bests)):
            r = self.compute(
                prev_best=prev_bests[i],
                current_best=current_bests[i],
                action_idx=actions[i] if actions is not None else None,
                state_info=state_infos[i] if state_infos else None,
                positions=positions_list[i] if positions_list else None,
                fitness_values=fitness_list[i] if fitness_list else None
            )
            rewards.append(r)
        return np.array(rewards, dtype=np.float32)

    def reset(self):
        self.danger_tracker.reset()
        self.coverage_tracker.reset()
        self.best_fitness = float('inf')
        for k in self.milestones: self.milestones[k] = False
        for q in self.stats.values(): q.clear()
    
    def soft_reset(self):
        """Episode 间重置，清除短期状态但保留长期统计(如需)"""
        self.reset()
        
    def get_stats(self) -> Dict:
        return {
            'num_danger_niches': len(self.danger_tracker.danger_niches),
            'num_niches': len(self.danger_tracker.danger_niches), # Alias
            'danger_coverage': self.danger_tracker._compute_coverage(),
            'best_fitness': self.best_fitness,
            'mean_reward': np.mean(self.stats['total']) if self.stats['total'] else 0.0,
            'milestones_achieved': sum(self.milestones.values())
        }

    def print_stats(self):
        s = self.get_stats()
        print(f"SafetyReward Stats: Niches={s['num_danger_niches']}, Cov={s['danger_coverage']:.2f}, BestFit={s['best_fitness']:.4f}")
