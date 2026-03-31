"""
niche_pso.py - 状态感知小生境PSO

整合所有改进组件：
1. 进化状态分类器 - 识别当前搜索阶段
2. 状态感知动作空间 - 8个简化动作
3. 聚类多样性奖励 - 密集学习信号
4. 增强NDM算子 - 跳出局部最优

设计目标：
- 让RL Agent能够收敛学习
- 提供清晰的状态-动作-奖励信号
- 支持多失效域/多模态搜索
"""

import numpy as np
import torch
from typing import Optional, List, Dict, Tuple
from collections import deque

# 本地导入（支持多种导入方式）
try:
    from .evo_state import (
        EvolutionaryState, EvolutionaryStateClassifier, 
        SubgroupStateTracker, StateInfo
    )
    from .action_space import StateAwareActionSpace
    from .reward import SafetyRewardCalculator
except ImportError:
    from evo_state import (
        EvolutionaryState, EvolutionaryStateClassifier,
        SubgroupStateTracker, StateInfo
    )
    from action_space import StateAwareActionSpace
    from reward import SafetyRewardCalculator

# 类型提示
try:
    from ..envs.base_env import BaseOptEnv
    from ..rl_core.dqn_agent import DoubleDQNAgent
except (ImportError, ValueError):
    try:
        from envs.base_env import BaseOptEnv
        from rl_core.dqn_agent import DoubleDQNAgent
    except ImportError:
        BaseOptEnv = None
        DoubleDQNAgent = None

# 自动检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StateAwareNichePSO:
    """
    状态感知小生境PSO (增强版)
    
    核心改进：
    1. 集成进化状态分类器，自动识别搜索阶段
    2. 简化动作空间（8个），降低学习难度
    3. 聚类多样性奖励，提供密集反馈
    4. 增强NDM算子，支持跳出局部最优
    
    状态特征 (增强版 - 28维)：
    - 进化状态 One-Hot (4维)
    - 适应度统计 + 正弦嵌入 (8维): CV_f嵌入(4) + 改进速率嵌入(4)
    - 多样性特征 (4维): 归一化特征
    - 时间嵌入 (5维): 进度 + sin/cos编码 + 阶段标志
    - 交互特征 (4维): 归一化距离
    - [NEW] 子种群感知特征 (3维): 排名, 相对适应度, 危险标志
    总计：28维
    """
    
    # 状态维度 (增强版 + Age + Subgroup Features)
    STATE_DIM = 28
    
    def __init__(self,
                 env: BaseOptEnv,
                 agent: Optional[DoubleDQNAgent] = None,
                 num_particles: int = 200,
                 num_subgroups: int = 10,
                 max_iterations: int = 100,
                 init_niche_radius: float = 0.2,
                 danger_threshold: float = -0.3,
                 use_gpu: bool = True,
                 enable_restart: bool = True,
                 restart_patience: int = 5,
                 task_type: str = 'unimodal',
                 action_interval: int = 10,
                 action_smoothing_factor: float = 0.6,
                 reward_calculator_class=None,
                 use_subgroup_features: bool = True):
        """
        Args:
            env: 优化环境
            agent: RL Agent（训练模式）或 None（固定参数模式）
            num_particles: 总粒子数
            num_subgroups: 子种群数量
            max_iterations: 最大迭代次数
            init_niche_radius: 初始小生境半径
            danger_threshold: 危险阈值
            use_gpu: 是否使用GPU
            enable_restart: 是否启用重启机制
            restart_patience: 停滞多少次后重启
            task_type: 任务类型 'unimodal' 或 'multimodal'
            reward_calculator_class: 自定义奖励计算器类 (默认 SafetyRewardCalculator)
            use_subgroup_features: 是否使用子种群感知特征 (default: True)
        """
        self.env = env
        self.agent = agent
        
        self.num_particles = num_particles
        self.num_subgroups = num_subgroups
        self.particles_per_group = num_particles // num_subgroups
        self.max_iterations = max_iterations
        self.init_niche_radius = init_niche_radius
        self.danger_threshold = danger_threshold
        
        # 重启机制
        self.enable_restart = enable_restart
        self.restart_patience = restart_patience
        
        # 任务类型
        self.task_type = task_type
        
        # 策略执行区间与平滑因子
        self.action_interval = action_interval
        self.action_smoothing_factor = action_smoothing_factor
        self.current_actions = None
        
        # 特征配置
        self.use_subgroup_features = use_subgroup_features
        self.reward_calculator_class = reward_calculator_class or SafetyRewardCalculator
        
        # Macro Command 状态缓存 (用于存储 (S_t, A_t))
        self.stored_states = None
        self.accumulated_rewards = None
        
        # 设备
        self.device = device if use_gpu and torch.cuda.is_available() else torch.device('cpu')
        
        # ===== 新增核心组件 =====
        # 进化状态追踪器
        self.state_tracker = SubgroupStateTracker(num_subgroups)
        
        # 状态感知动作空间
        self.action_space = StateAwareActionSpace()
        self.num_actions = self.action_space.get_num_actions()  # 8
        self.state_dim = self.STATE_DIM
        
        # 奖励计算器（延迟初始化，需要bounds和dim）
        self.reward_calculator = None
        
        # print(f"[StateAwareNichePSO] Initialized")
        # print(f"  State dim: {self.state_dim}, Actions: {self.num_actions}")
        # print(f"  Particles: {num_particles}, Subgroups: {num_subgroups}")
        # print(f"  Device: {self.device}")
        
        # Agent类型检测
        self._agent_action_type = 'discrete'  # 默认离散
        
        # ===== 粒子数据 =====
        self._dim = None
        self._bounds = None
        
        self.all_particles = None
        self.all_velocities = None
        self.all_pbest_positions = None
        self.all_pbest_fitness = None
        self.all_raw_fitness = None
        
        # 子种群参数
        self.group_gbest_positions = None
        self.group_gbest_fitness = None
        self.group_w = None
        self.group_c1 = None
        self.group_c2 = None
        self.group_niche_radius = None
        self.group_stagnation = None
        self.group_prev_best = None
        self.group_prev_diversity = None
        self.group_restart_counter = None
        self.group_ages = None  # 新增：子种群年龄
        
        # 当前进化状态
        self.current_state_infos: List[StateInfo] = []
        
        # 危险点收集
        self.global_hazardous_pool = []
        
        # 统计
        self.mean_rewards = []
        self.iteration_count = 0
        
        # 兼容接口
        self.subgroups = []
    
    def init_subgroups(self, reset_trackers: bool = True):
        """
        初始化所有子种群
        
        Args:
            reset_trackers: 是否重置奖励追踪器（切换函数时设为False）
        """
        bounds = self.env.get_bounds()
        self._bounds = bounds
        dim = self.env.get_dim()
        self._dim = dim
        
        total_particles = self.num_particles
        
        # 初始化粒子位置
        self.all_particles = torch.rand(total_particles, dim, device=self.device)
        self.all_particles = bounds[0] + (bounds[1] - bounds[0]) * self.all_particles
        
        # 初始化速度
        v_range = (bounds[1] - bounds[0]) * 0.1
        self.all_velocities = (torch.rand(total_particles, dim, device=self.device) - 0.5) * v_range
        
        # 初始化个体最优
        self.all_pbest_positions = self.all_particles.clone()
        self.all_pbest_fitness = torch.full((total_particles,), float('inf'), device=self.device)
        
        # 初始评估
        self._evaluate_fitness_batch()
        self.all_pbest_fitness = self.all_raw_fitness.clone()
        
        # 初始化子种群参数
        self.group_gbest_positions = torch.zeros(self.num_subgroups, dim, device=self.device)
        self.group_gbest_fitness = torch.full((self.num_subgroups,), float('inf'), device=self.device)
        self.group_w = torch.full((self.num_subgroups,), 0.7, device=self.device)
        self.group_c1 = torch.full((self.num_subgroups,), 1.5, device=self.device)
        self.group_c2 = torch.full((self.num_subgroups,), 1.5, device=self.device)
        self.group_niche_radius = torch.full((self.num_subgroups,), self.init_niche_radius, device=self.device)
        self.group_stagnation = torch.zeros(self.num_subgroups, dtype=torch.int32, device=self.device)
        self.group_prev_best = torch.full((self.num_subgroups,), float('inf'), device=self.device)
        self.group_prev_diversity = torch.full((self.num_subgroups,), 0.5, device=self.device)
        self.group_prev_diversity = torch.full((self.num_subgroups,), 0.5, device=self.device)
        self.group_restart_counter = torch.zeros(self.num_subgroups, dtype=torch.int32, device=self.device)
        self.group_ages = torch.zeros(self.num_subgroups, dtype=torch.int32, device=self.device)
        
        # 更新各组 gBest
        for g in range(self.num_subgroups):
            start = g * self.particles_per_group
            end = (g + 1) * self.particles_per_group
            group_fitness = self.all_raw_fitness[start:end]
            best_idx = group_fitness.argmin()
            self.group_gbest_fitness[g] = group_fitness[best_idx]
            self.group_gbest_positions[g] = self.all_particles[start + best_idx]
            self.group_prev_best[g] = self.group_gbest_fitness[g]
        
        # 初始化奖励计算器（首次）或选择性重置
        if self.reward_calculator is None:
            self.reward_calculator = self.reward_calculator_class(
                bounds=bounds,
                dim=dim,
                niche_radius=self.init_niche_radius * 2,
                danger_threshold=self.danger_threshold
            )
        elif reset_trackers:
            # 完全重置（新训练开始）
            self.reward_calculator.reset()
        # 否则保持追踪器状态（切换函数时）
        
        # 初始化状态追踪器
        self.state_tracker.reset()
        self.current_state_infos = [None] * self.num_subgroups
        
        # 初始化当前动作和缓存
        # 检测Agent类型
        if self.agent is not None and hasattr(self.agent, 'action_type'):
            self._agent_action_type = self.agent.action_type
        elif self.agent is not None and hasattr(self.agent, 'action_dim'):
            # PPOAgent 输出4维连续动作
            self._agent_action_type = 'continuous'
        else:
            self._agent_action_type = 'discrete'
        
        # 根据动作类型初始化
        if self._agent_action_type == 'continuous':
            # 连续动作：初始化为默认PSO参数
            if self.current_actions is None or len(self.current_actions) != self.num_subgroups:
                self.current_actions = [np.array([0.7, 1.5, 1.5, 1.0], dtype=np.float32) for _ in range(self.num_subgroups)]
            # 连续动作参数存储 (4维: w, c1, c2, velocity_scale)
            self.current_probs = np.zeros((self.num_subgroups, 4), dtype=np.float32)
            self.current_probs[:, 0] = 0.7  # w
            self.current_probs[:, 1] = 1.5  # c1
            self.current_probs[:, 2] = 1.5  # c2
            self.current_probs[:, 3] = 1.0  # velocity_scale
        else:
            # 离散动作
            if self.current_actions is None or len(self.current_actions) != self.num_subgroups:
                self.current_actions = [2] * self.num_subgroups
            # 离散动作概率分布 (6维)
            self.current_probs = np.zeros((self.num_subgroups, self.num_actions), dtype=np.float32)
            self.current_probs[:, 2] = 1.0  # 默认 Balanced
        
        self.stored_states = [None] * self.num_subgroups
        self.accumulated_rewards = np.zeros(self.num_subgroups, dtype=np.float32)
        
        # 并行轨迹缓存 (每组独立的 buffer list)
        # 存储格式: [(s, a, r, s', log_prob, value), ...]
        self.group_trajectory_buffers = [[] for _ in range(self.num_subgroups)]
        
        # 暂存动作的额外信息 (log_prob, value)
        self.stored_action_extras = [None] * self.num_subgroups
        
        # 构建兼容接口
        self._build_subgroups_compat()
        
        self.iteration_count = 0
        self.global_hazardous_pool = []
        self.mean_rewards = []
    
    def soft_reinit(self):
        """
        软重置：重新初始化粒子但保持奖励追踪器状态
        
        用于切换优化函数时保持小生境和覆盖率的累积统计
        """
        self.init_subgroups(reset_trackers=False)
    
    def _build_subgroups_compat(self):
        """构建兼容旧接口的 subgroups 列表"""
        self.subgroups = []
        for g in range(self.num_subgroups):
            start = g * self.particles_per_group
            end = (g + 1) * self.particles_per_group
            
            self.subgroups.append({
                'particles': self.all_particles[start:end].cpu().numpy(),
                'velocities': self.all_velocities[start:end].cpu().numpy(),
                'fitness': self.all_raw_fitness[start:end].cpu().numpy(),
                'global_best_position': self.group_gbest_positions[g].cpu().numpy(),
                'global_best_fitness': self.group_gbest_fitness[g].item(),
                'w': self.group_w[g].item(),
                'c1': self.group_c1[g].item(),
                'c2': self.group_c2[g].item(),
                'niche_radius': self.group_niche_radius[g].item(),
                'stagnation_counter': self.group_stagnation[g].item(),
                'last_action': self.current_actions[g],
                'last_probs': self.current_probs[g].copy() if self.current_probs is not None else None,
            })
    
    def _sinusoidal_embedding(self, x: float, freq_bands: int = 2) -> np.ndarray:
        """
        正弦高维嵌入 - 增强对微小变化的感知
        
        Args:
            x: 输入标量 (应在 [-1, 1] 范围内)
            freq_bands: 频率带数量
            
        Returns:
            embedding: (2 * freq_bands,) 维嵌入向量
        """
        x = np.clip(x, -1, 1)
        freqs = [2**i for i in range(freq_bands)]
        embedding = []
        for f in freqs:
            embedding.append(np.sin(np.pi * f * x))
            embedding.append(np.cos(np.pi * f * x))
        return np.array(embedding, dtype=np.float32)
    
    def _get_state(self, group_idx: int) -> np.ndarray:
        """
        获取增强版状态特征（25维）
        
        特征组成：
        - 进化状态 One-Hot (4维)
        - 适应度统计 + 正弦嵌入 (8维): CV_f嵌入(4) + 改进速率嵌入(4)
        - 多样性特征 (4维): 位置多样性/速度比/聚集度/半径比
        - 时间嵌入 (5维): progress + sin(2πp) + cos(2πp) + early + late
        - 交互特征 (4维): min_dist, mean_dist, overlap, isolated
        
        关键改进：
        1. 无量纲化：使用变异系数、比值等替代绝对值
        2. 正弦嵌入：增强对微小变化的感知
        3. 时间感知：显式进度编码解决非平稳性
        """
        start = group_idx * self.particles_per_group
        end = (group_idx + 1) * self.particles_per_group
        
        positions = self.all_particles[start:end].cpu().numpy()
        fitness = self.all_raw_fitness[start:end].cpu().numpy()
        velocities = self.all_velocities[start:end].cpu().numpy()
        gbest_pos = self.group_gbest_positions[group_idx].cpu().numpy()
        gbest_fit = self.group_gbest_fitness[group_idx].item()
        stagnation = self.group_stagnation[group_idx].item()
        prev_best = self.group_prev_best[group_idx].item()
        
        features = []
        
        # ===== 1. 进化状态 One-Hot (4维) =====
        state_info = self.current_state_infos[group_idx]
        if state_info is not None:
            one_hot = np.zeros(4, dtype=np.float32)
            one_hot[int(state_info.state)] = 1.0
            features.extend(one_hot)
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # ===== 2. 适应度统计 + 正弦嵌入 (8维) =====
        fit_mean = np.mean(fitness)
        fit_std = np.std(fitness)
        
        # 变异系数 (无量纲化) - 使用 tanh 归一化到 [-1, 1]
        cv_f = fit_std / (np.abs(fit_mean) + 1e-8)
        cv_normalized = np.tanh(cv_f)  # 归一化到 [-1, 1]
        
        # 适应度改进速率 (无量纲化)
        if prev_best != 0 and not np.isinf(prev_best):
            improvement_rate = (prev_best - gbest_fit) / (np.abs(prev_best) + 1e-8)
        else:
            improvement_rate = 0.0
        improvement_normalized = np.tanh(improvement_rate * 10)  # 放大后归一化
        
        # 正弦嵌入: CV_f (4维) + 改进速率 (4维)
        cv_embedding = self._sinusoidal_embedding(cv_normalized, freq_bands=2)
        improvement_embedding = self._sinusoidal_embedding(improvement_normalized, freq_bands=2)
        features.extend(cv_embedding)
        features.extend(improvement_embedding)
        
        # ===== 3. 多样性特征 (4维) - 完全无量纲化 =====
        pos_range = self._bounds[1] - self._bounds[0]
        
        # 位置多样性：归一化标准差 (0-1)
        positions_norm = (positions - self._bounds[0]) / (pos_range + 1e-8)
        pos_diversity = positions_norm.std(axis=0).mean()  # 已在 [0, 0.5] 范围
        
        # 速度比：当前速度 / 最大速度
        v_max = pos_range * 0.2
        vel_norms = np.linalg.norm(velocities, axis=1)
        velocity_ratio = vel_norms.mean() / (v_max * np.sqrt(self._dim) + 1e-8)
        
        # 聚集度：到质心的平均距离 / 搜索空间对角线
        centroid = positions_norm.mean(axis=0)
        dist_to_centroid = np.linalg.norm(positions_norm - centroid, axis=1).mean()
        diagonal = np.sqrt(self._dim)  # 归一化空间对角线
        clustering_ratio = dist_to_centroid / diagonal
        
        features.extend([
            np.clip(pos_diversity * 2, 0, 1),      # 缩放到 [0, 1]
            np.clip(velocity_ratio, 0, 1),
            np.clip(clustering_ratio, 0, 1)
        ])
        
        # ===== 4. 时间嵌入 (5维) =====
        progress = self.iteration_count / (self.max_iterations + 1e-8)
        
        features.extend([
            progress,                               # 线性进度
            np.sin(2 * np.pi * progress),           # 周期性嵌入 sin
            np.cos(2 * np.pi * progress),           # 周期性嵌入 cos
            1.0 if progress < 0.3 else 0.0,         # 早期阶段标志
            1.0 if progress > 0.7 else 0.0,         # 后期阶段标志
        ])
        
        # ===== 5. 交互特征 (4维) - 无量纲化 =====
        all_gbest = self.group_gbest_positions.cpu().numpy()
        all_gbest_norm = (all_gbest - self._bounds[0]) / (pos_range + 1e-8)
        gbest_pos_norm = (gbest_pos - self._bounds[0]) / (pos_range + 1e-8)
        
        if len(all_gbest) > 1:
            other_gbests_norm = np.delete(all_gbest_norm, group_idx, axis=0)
            dists = np.linalg.norm(other_gbests_norm - gbest_pos_norm, axis=1)
            min_dist = dists.min() / diagonal  # 归一化
            mean_dist = dists.mean() / diagonal
            overlap = np.sum(dists < 2 * self.init_niche_radius) / len(other_gbests_norm)
            isolated = 1.0 if min_dist > 0.3 else 0.0
        else:
            min_dist, mean_dist, overlap, isolated = 0.5, 0.5, 0.0, 0.0
        
        features.extend([
            np.clip(min_dist, 0, 1),
            np.clip(mean_dist, 0, 1),
            np.clip(overlap, 0, 1),
            isolated
        ])
        
        # ===== 6. 年龄特征 (1维) =====
        # 归一化年龄 (主要区分刚重启的种群)
        age = self.group_ages[group_idx].item()
        # 使用 sigmoid 压缩，让 0-20 代变化最明显
        age_feature = np.tanh(age / 20.0)
        
        features.append(age_feature)
        
        # ===== 7. 子种群感知特征 (3维) - [NEW] =====
        if self.use_subgroup_features:
            # 1. 排名 (0=Best, 1=Worst)
            all_gbest_fitness = self.group_gbest_fitness.cpu().numpy()
            sorted_indices = np.argsort(all_gbest_fitness)
            rank = np.where(sorted_indices == group_idx)[0][0]
            normalized_rank = rank / (self.num_subgroups - 1 + 1e-8)
            
            # 2. 相对适应度 (0=GlobalBest, 1=GlobalWorst)
            global_best = all_gbest_fitness.min()
            global_worst = all_gbest_fitness.max()
            if global_worst - global_best > 1e-8:
                relative_fitness = (gbest_fit - global_best) / (global_worst - global_best)
            else:
                relative_fitness = 0.5
            
            # 3. 危险区域标志
            is_in_danger_zone = 1.0 if gbest_fit < self.danger_threshold else 0.0
            
            features.extend([normalized_rank, np.clip(relative_fitness, 0, 1), is_in_danger_zone])
        
        # Pad to fixed size if needed
        current_len = len(features)
        if current_len < self.STATE_DIM:
             features.extend([0.0] * (self.STATE_DIM - current_len))
        
        return np.array(features, dtype=np.float32)
    
    def _get_diversity(self, group_idx: int) -> float:
        """计算子种群多样性（归一化）"""
        start = group_idx * self.particles_per_group
        end = (group_idx + 1) * self.particles_per_group
        
        positions = self.all_particles[start:end]
        pos_range = self._bounds[1] - self._bounds[0]
        # 归一化后计算标准差
        positions_norm = (positions - self._bounds[0]) / (pos_range + 1e-8)
        pos_std = positions_norm.std(dim=0).mean()
        
        return pos_std.item()  # 已经归一化，直接返回
    
    def _evaluate_fitness_batch(self):
        """批量评估适应度"""
        particles_np = self.all_particles.cpu().numpy()
        fitness_np, _ = self.env.evaluate(particles_np)
        self.all_raw_fitness = torch.tensor(fitness_np, dtype=torch.float32, device=self.device)
        
        # 收集危险点
        hazardous_mask = fitness_np < self.danger_threshold
        if np.any(hazardous_mask):
            self.global_hazardous_pool.extend(particles_np[hazardous_mask].tolist())
        
        # 更新个体最优
        improved = self.all_raw_fitness < self.all_pbest_fitness
        self.all_pbest_positions[improved] = self.all_particles[improved]
        self.all_pbest_fitness[improved] = self.all_raw_fitness[improved]
    
    def _update_group_bests(self):
        """更新各组全局最优"""
        for g in range(self.num_subgroups):
            start = g * self.particles_per_group
            end = (g + 1) * self.particles_per_group
            
            group_fitness = self.all_raw_fitness[start:end]
            best_idx = group_fitness.argmin()
            best_fitness = group_fitness[best_idx]
            
            if best_fitness < self.group_gbest_fitness[g]:
                self.group_gbest_fitness[g] = best_fitness
                self.group_gbest_positions[g] = self.all_particles[start + best_idx]
                self.group_stagnation[g] = 0
            else:
                self.group_stagnation[g] += 1
    
    def _apply_action(self, group_idx: int, action_idx: int):
        """应用动作到子种群 (增强版)"""
        state_info = self.current_state_infos[group_idx]
        stagnation = self.group_stagnation[group_idx].item()
        diversity = self._get_diversity(group_idx)
        
        params = self.action_space.get_action_params(
            action_idx, state_info, stagnation, diversity
        )
        
        # 平滑更新参数 (Scheme B)
        alpha = self.action_smoothing_factor
        self.group_w[group_idx] = (1 - alpha) * self.group_w[group_idx] + alpha * params['w']
        self.group_c1[group_idx] = (1 - alpha) * self.group_c1[group_idx] + alpha * params['c1']
        self.group_c2[group_idx] = (1 - alpha) * self.group_c2[group_idx] + alpha * params['c2']
        
        # 移除动态半径调整
        # new_radius = self.init_niche_radius * params['radius_scale'] ...
        
        # 速度调整（修复：使用基础v_max缩放，避免累乘导致速度爆炸）
        velocity_scale = params.get('velocity_scale', 1.0)
        base_v_max = (self._bounds[1] - self._bounds[0]) * 0.2
        actual_v_max = base_v_max * velocity_scale
        
        start = group_idx * self.particles_per_group
        end = (group_idx + 1) * self.particles_per_group
        
        # 将速度约束到新的范围内
        self.all_velocities[start:end] = self.all_velocities[start:end].clamp(-actual_v_max, actual_v_max)
        
        # NDM触发
        if params.get('trigger_ndm', False):
            self._apply_enhanced_ndm(group_idx)
    
    def _apply_continuous_action(self, group_idx: int, action: np.ndarray):
        """
        应用连续动作到子种群
        
        Args:
            group_idx: 子种群索引
            action: 连续动作数组 [w, c1, c2, velocity_scale]
        """
        w, c1, c2, velocity_scale = action[0], action[1], action[2], action[3]
        
        # 平滑更新参数 (Scheme B)
        alpha = self.action_smoothing_factor
        self.group_w[group_idx] = (1 - alpha) * self.group_w[group_idx] + alpha * w
        self.group_c1[group_idx] = (1 - alpha) * self.group_c1[group_idx] + alpha * c1
        self.group_c2[group_idx] = (1 - alpha) * self.group_c2[group_idx] + alpha * c2
        
        # 速度调整（修复：使用基础v_max缩放，避免累乘导致速度爆炸）
        base_v_max = (self._bounds[1] - self._bounds[0]) * 0.2
        actual_v_max = base_v_max * velocity_scale
        
        start = group_idx * self.particles_per_group
        end = (group_idx + 1) * self.particles_per_group
        
        # 将速度约束到新的范围内（而非累乘）
        self.all_velocities[start:end] = self.all_velocities[start:end].clamp(-actual_v_max, actual_v_max)
    
    def _apply_enhanced_ndm(self, group_idx: int):
        """
        增强版邻域差分变异 (NDM)
        
        改进：应用到最差30%粒子，而不仅仅是最差的一个
        """
        start = group_idx * self.particles_per_group
        end = (group_idx + 1) * self.particles_per_group
        
        particles = self.all_particles[start:end]
        fitness = self.all_raw_fitness[start:end]
        center = self.group_gbest_positions[group_idx]
        
        # 选择最远和最近粒子
        distances = (particles - center).norm(dim=1)
        g_far = particles[distances.argmax()]
        g_near = particles[distances.argmin()]
        
        # 差分变异缩放因子 (随机化)
        F = 0.5 + 0.5 * torch.rand(1, device=self.device)
        
        # 应用到最差30%粒子
        num_mutate = max(1, int(self.particles_per_group * 0.3))
        worst_indices = fitness.argsort(descending=True)[:num_mutate]
        
        for idx in worst_indices:
            mutation = F * (g_far - g_near) + 0.1 * torch.randn(self._dim, device=self.device)
            new_position = (particles[idx] + mutation).clamp(
                self._bounds[0], self._bounds[1]
            )
            self.all_particles[start + idx] = new_position
        
        # 重置该组停滞计数
        self.group_stagnation[group_idx] = max(0, self.group_stagnation[group_idx] - 3)
    
    def _update_velocity_batch(self):
        """批量更新速度"""
        group_indices = torch.arange(self.num_particles, device=self.device) // self.particles_per_group
        
        w = self.group_w[group_indices].unsqueeze(1)
        c1 = self.group_c1[group_indices].unsqueeze(1)
        c2 = self.group_c2[group_indices].unsqueeze(1)
        
        r1 = torch.rand(self.num_particles, 1, device=self.device)
        r2 = torch.rand(self.num_particles, 1, device=self.device)
        
        gbest_expanded = self.group_gbest_positions[group_indices]
        
        cognitive = c1 * r1 * (self.all_pbest_positions - self.all_particles)
        social = c2 * r2 * (gbest_expanded - self.all_particles)
        
        self.all_velocities = w * self.all_velocities + cognitive + social
        
        v_max = (self._bounds[1] - self._bounds[0]) * 0.2
        self.all_velocities = self.all_velocities.clamp(-v_max, v_max)
    
    def _update_position_batch(self):
        """批量更新位置"""
        self.all_particles = (self.all_particles + self.all_velocities).clamp(
            self._bounds[0], self._bounds[1]
        )
    
    def _flush_group_buffer(self, group_idx: int, done: bool = True):
        """刷新单个组的轨迹缓存到Agent"""
        if not hasattr(self, 'group_trajectory_buffers'):
            return
        if self.agent is None or self.agent.frozen:
            return
            
        buf = self.group_trajectory_buffers[group_idx]
        if not buf:
            return
            
        for i, transition in enumerate(buf):
            state, action, reward, next_state, log_prob, value = transition
            is_last = (i == len(buf) - 1)
            d = done if is_last else False
            
            # 根据Agent类型选择不同的调用方式
            if self._agent_action_type == 'continuous':
                # PPO: 支持 done, log_prob, value 参数
                try:
                    self.agent.store_transition(state, action, reward, next_state,
                                              done=d, log_prob=log_prob, value=value)
                except TypeError:
                    # Fallback: 不带额外参数
                    self.agent.store_transition(state, action, reward, next_state)
            else:
                # DDQN: 只接受 (state, action, reward, next_state, done)
                try:
                    self.agent.store_transition(state, action, reward, next_state, done=d)
                except TypeError:
                    # Backward compatibility safely if old agent loaded
                    self.agent.store_transition(state, action, reward, next_state)
        
        buf.clear()
    
    def flush_experience_to_agent(self, done_all: bool = False):
        """
        将所有缓存的轨迹刷新到Agent缓冲区
        
        解决并行轨迹混合导致的GAE计算错误问题。
        必须在Episode结束时调用。
        
        Args:
            done_all: 是否标记所有最后一步为Done
        """
        if not hasattr(self, 'group_trajectory_buffers'):
            return
            
        for g in range(self.num_subgroups):
            self._flush_group_buffer(g, done=done_all)
    
    def _restart_subgroup(self, group_idx: int):
        """重启子种群"""
        # 重启前先刷新该组的轨迹缓存（标记为Done）
        self._flush_group_buffer(group_idx, done=True)
        
        start = group_idx * self.particles_per_group
        end = (group_idx + 1) * self.particles_per_group
        
        # 随机初始化新位置
        new_positions = torch.rand(self.particles_per_group, self._dim, device=self.device)
        new_positions = self._bounds[0] + (self._bounds[1] - self._bounds[0]) * new_positions
        
        v_range = (self._bounds[1] - self._bounds[0]) * 0.1
        new_velocities = (torch.rand(self.particles_per_group, self._dim, device=self.device) - 0.5) * v_range
        
        self.all_particles[start:end] = new_positions
        self.all_velocities[start:end] = new_velocities
        self.all_pbest_positions[start:end] = new_positions.clone()
        
        # 评估新位置
        particles_np = new_positions.cpu().numpy()
        fitness_np, _ = self.env.evaluate(particles_np)
        new_fitness = torch.tensor(fitness_np, dtype=torch.float32, device=self.device)
        
        self.all_pbest_fitness[start:end] = new_fitness
        self.all_raw_fitness[start:end] = new_fitness
        
        # 更新全局最优
        best_idx = new_fitness.argmin()
        self.group_gbest_fitness[group_idx] = new_fitness[best_idx]
        self.group_gbest_positions[group_idx] = new_positions[best_idx]
        
        # 重置参数
        self.group_w[group_idx] = 0.7
        self.group_c1[group_idx] = 1.5
        self.group_c2[group_idx] = 1.5
        self.group_niche_radius[group_idx] = self.init_niche_radius
        self.group_stagnation[group_idx] = 0
        self.group_restart_counter[group_idx] += 1
        self.group_ages[group_idx] = 0  # 重置年龄
    
    def run_one_iteration(self) -> List[float]:
        """运行一次迭代"""
        progress = self.iteration_count / (self.max_iterations + 1e-8)
        
        # 保存上一轮状态
        prev_bests = self.group_gbest_fitness.clone()
        prev_diversities = torch.tensor(
            [self._get_diversity(g) for g in range(self.num_subgroups)],
            device=self.device
        )
        
        # 1. 更新进化状态
        for g in range(self.num_subgroups):
            start = g * self.particles_per_group
            end = (g + 1) * self.particles_per_group
            
            positions = self.all_particles[start:end].cpu().numpy()
            gbest_pos = self.group_gbest_positions[g].cpu().numpy()
            stagnation = self.group_stagnation[g].item()
            
            state_info = self.state_tracker.update(g, positions, gbest_pos, stagnation)
            self.current_state_infos[g] = state_info
        
        # 2. 获取状态和选择动作 (Macro Command Mode)
        # 逻辑：
        # - 在 Decision Step (t):
        #   1. 如果有缓存的 (S_{t-K}, A_{t-K})，结合当前 S_t 和累积奖励 R_accum，存储 transition
        #   2. 观测当前 S_t，选择新动作 A_t
        #   3. 缓存 S_t, A_t，重置 R_accum
        # - 在 Execution Step:
        #   1. 维持 A_t
        
        is_decision_step = (self.iteration_count % self.action_interval == 0)
        current_step_states = []  # 暂存当前步计算出的状态
        
        # 先计算所有组的当前状态 S_t
        for g in range(self.num_subgroups):
            state = self._get_state(g)
            current_step_states.append(state)
        
        if is_decision_step:
            # === A. 存储上一轮的 Macro Transition (S_{t-K}, A_{t-K}, R_sum, S_t) ===
            if self.agent is not None and not self.agent.frozen and self.stored_states[0] is not None:
                for g in range(self.num_subgroups):
                    # 只有当上一轮确实有动作时才存储
                    if self.stored_states[g] is not None:
                        # 获取缓存的 log_prob 和 value (从上一轮决策时保存)
                        extras = self.stored_action_extras[g]
                        log_prob = extras['log_prob'] if extras else None
                        value = extras['value'] if extras else None
                        
                        # 将Transition存入该组的Buffer，而不是直接存入Agent
                        # 格式: (state, action, reward, next_state, log_prob, value)
                        if hasattr(self, 'group_trajectory_buffers'):
                            self.group_trajectory_buffers[g].append((
                                self.stored_states[g],
                                self.current_actions[g],
                                self.accumulated_rewards[g],
                                current_step_states[g],
                                log_prob,
                                value
                            ))
            
            # === B. 决策新动作 A_t ===
            for g in range(self.num_subgroups):
                state = current_step_states[g]
                
                if self.agent is not None:
                    # 获取动作及其信息
                    action_result = self.agent.choose_action(state)
                    
                    # 兼容处理：检查返回的是元组还是仅动作索引
                    if isinstance(action_result, tuple):
                        action, action_info = action_result
                    else:
                        action = action_result
                        action_info = None
                    
                    # 提取 log_prob 和 value (用于PPO)
                    log_prob = None
                    value = None
                    if isinstance(action_info, dict):
                        log_prob = action_info.get('log_prob')
                        value = action_info.get('value')
                    
                    # 缓存动作信息供下一步存储使用
                    if hasattr(self, 'stored_action_extras'):
                        self.stored_action_extras[g] = {'log_prob': log_prob, 'value': value}
                    
                    # 检测Agent类型（仅在首次时检测）
                    if self.iteration_count == 0 and g == 0:
                        if hasattr(self.agent, 'action_type'):
                            self._agent_action_type = self.agent.action_type
                        elif isinstance(action, np.ndarray) and action.ndim > 0:
                            self._agent_action_type = 'continuous'
                        else:
                            self._agent_action_type = 'discrete'
                else:
                    action = 0 # 无Agent模式: 默认使用 Wide Scout (0)
                    action_info = None
                    if hasattr(self, 'stored_action_extras'):
                        self.stored_action_extras[g] = None
                
                self.current_actions[g] = action
                
                # 存储该组的动作信息（用于可视化）
                if self._agent_action_type == 'continuous':
                    # 连续动作：存储动作向量本身 (4维: w, c1, c2, velocity_scale)
                    if isinstance(action, np.ndarray):
                        self.current_probs[g] = action[:4].copy()  # 确保只取4维
                    else:
                        self.current_probs[g] = np.array([0.7, 1.5, 1.5, 1.0], dtype=np.float32)
                elif action_info is not None:
                    # 离散动作：存储概率分布
                    if isinstance(action_info, np.ndarray) and len(action_info) == self.num_actions:
                        self.current_probs[g] = action_info
                    elif isinstance(action_info, dict) and 'mean' in action_info:
                        self.current_probs[g] = action_info['mean']
                    else:
                        one_hot = np.zeros(self.num_actions, dtype=np.float32)
                        one_hot[int(action)] = 1.0
                        self.current_probs[g] = one_hot
                else:
                    # Fallback: 构造 One-Hot
                    one_hot = np.zeros(self.num_actions, dtype=np.float32)
                    one_hot[int(action)] = 1.0
                    self.current_probs[g] = one_hot
                
                # === C. 缓存 S_t 和 A_t，重置累积奖励 ===
                self.stored_states[g] = state
                self.accumulated_rewards[g] = 0.0
        
        # === D. 执行 (每步都应用当前动作) ===
        for g in range(self.num_subgroups):
            if self._agent_action_type == 'continuous':
                # 连续动作：直接应用参数
                self._apply_continuous_action(g, self.current_actions[g])
            else:
                # 离散动作：通过ActionSpace映射
                self._apply_action(g, self.current_actions[g])
        
        # 3. PSO更新
        self._update_velocity_batch()
        self._update_position_batch()
        self._evaluate_fitness_batch()
        self._update_group_bests()
        
        # 4. 检查重启条件
        if self.enable_restart:
            for g in range(self.num_subgroups):
                if self.group_stagnation[g] >= self.restart_patience:
                    self._restart_subgroup(g)
        
        # 5. 计算奖励 (Dense Reward)
        current_diversities = torch.tensor(
            [self._get_diversity(g) for g in range(self.num_subgroups)],
            device=self.device
        )
        
        # 获取各组位置用于奖励计算
        positions_list = []
        fitness_values_list = []  # 新增：每个粒子的适应度
        for g in range(self.num_subgroups):
            start = g * self.particles_per_group
            end = (g + 1) * self.particles_per_group
            positions_list.append(self.all_particles[start:end].cpu().numpy())
            fitness_values_list.append(self.all_raw_fitness[start:end].cpu().numpy())
        
        # 尝试传递 fitness_values_list（PPO V3需要）
        try:
            rewards = self.reward_calculator.compute_batch(
                prev_bests=prev_bests.cpu().numpy(),
                current_bests=self.group_gbest_fitness.cpu().numpy(),
                prev_diversities=prev_diversities.cpu().numpy(),
                current_diversities=current_diversities.cpu().numpy(),
                stagnation_counts=self.group_stagnation.cpu().numpy(),
                actions=np.array(self.current_actions),
                progress=progress,
                state_infos=self.current_state_infos,
                positions_list=positions_list,
                gbest_positions=self.group_gbest_positions.cpu().numpy(),
                func_name=self.env.get_current_function_name() if hasattr(self.env, 'get_current_function_name') else None,
                fitness_list=fitness_values_list  # Use fitness_list to match SafetyRewardCalculator
            )
        except TypeError:
            # Fallback: 不支持 fitness_values_list 的旧版本
            rewards = self.reward_calculator.compute_batch(
                prev_bests=prev_bests.cpu().numpy(),
                current_bests=self.group_gbest_fitness.cpu().numpy(),
                prev_diversities=prev_diversities.cpu().numpy(),
                current_diversities=current_diversities.cpu().numpy(),
                stagnation_counts=self.group_stagnation.cpu().numpy(),
                actions=np.array(self.current_actions),
                progress=progress,
                state_infos=self.current_state_infos,
                positions_list=positions_list,
                gbest_positions=self.group_gbest_positions.cpu().numpy(),
                func_name=self.env.get_current_function_name() if hasattr(self.env, 'get_current_function_name') else None
            )
        
        # === E. 累积奖励 ===
        # 将单步奖励累积到缓冲区
        rewards = np.tanh(rewards) # 数值压缩，防止爆炸
        self.accumulated_rewards += rewards
        
        # 6. 存储经验 (改为在 Decision Step 批量存)
        # (已移动到上方 Decision Step 块)
        
        # 更新兼容接口
        self._build_subgroups_compat()
        
        self.iteration_count += 1
        self.group_ages += 1 # 更新子种群年龄
        self.mean_rewards.append(np.mean(rewards))
        
        return rewards.tolist()
    
    def run(self, verbose: bool = True) -> np.ndarray:
        """运行完整优化"""
        self.init_subgroups()
        
        if verbose:
            print(f"\nStarting StateAwareNichePSO optimization...")
            print(f"  State dim: {self.state_dim}, Actions: {self.num_actions}")
        
        try:
            from tqdm import tqdm
            iterator = tqdm(range(self.max_iterations))
        except ImportError:
            iterator = range(self.max_iterations)
        
        for i in iterator:
            rewards = self.run_one_iteration()
            
            if self.agent is not None and not self.agent.frozen:
                self.agent.update_q_values()
                if i % 10 == 0:
                    self.agent.soft_update_target_network(tau=0.005)
                self.agent.decay_epsilon()
            
            if verbose and i % 50 == 0:
                best = self.group_gbest_fitness.min().item()
                niches = self.reward_calculator.niche_tracker.get_num_niches()
                print(f"Iter {i}: Best={best:.6f}, Niches={niches}, "
                      f"Pool={len(self.global_hazardous_pool)}")
        
        # 返回去重后的危险点
        if len(self.global_hazardous_pool) > 0:
            hazards = np.array(self.global_hazardous_pool)
            _, unique_idx = np.unique(np.round(hazards, decimals=4), axis=0, return_index=True)
            return hazards[unique_idx]
        else:
            return np.array([])
    
    def get_training_stats(self) -> dict:
        """获取训练统计"""
        return {
            'mean_rewards': self.mean_rewards,
            'hazardous_pool_size': len(self.global_hazardous_pool),
            'evaluation_count': self.env.get_evaluation_count(),
            'iterations': self.iteration_count,
            'state_dim': self.state_dim,
            'num_actions': self.num_actions,
            'num_niches': self.reward_calculator.niche_tracker.get_num_niches(),
            'coverage_rate': self.reward_calculator.coverage_tracker.get_coverage_rate()
        }


# ===== 工具函数 =====

def get_state_aware_state_dim() -> int:
    """获取状态感知PSO的状态维度"""
    return StateAwareNichePSO.STATE_DIM

def get_state_aware_action_dim() -> int:
    """获取状态感知动作空间维度"""
    return 6
