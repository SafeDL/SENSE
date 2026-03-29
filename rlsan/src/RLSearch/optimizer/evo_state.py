"""
state_classifier.py - 进化状态分类器

参考文献：Li et al. (2023) "Reinforcement learning-based particle swarm optimization 
with neighborhood differential mutation strategy"

将PSO搜索过程分为四种进化状态：
- S1 (Exploration): 种群分散，多样性高
- S2 (Exploitation): 向全局最优收敛
- S3 (Convergence): 高度聚集，接近最优
- S4 (Jump-out): 停滞状态，需要逃离局部最优

每种状态对应不同的参数控制策略，RL Agent学习何时遵循建议，何时探索其他策略。
"""

import numpy as np
from typing import Tuple, Dict, NamedTuple
from enum import IntEnum


class EvolutionaryState(IntEnum):
    """进化状态枚举"""
    EXPLORATION = 0    # S1: 探索阶段
    EXPLOITATION = 1   # S2: 开发阶段
    CONVERGENCE = 2    # S3: 收敛阶段
    JUMP_OUT = 3       # S4: 跳出阶段


class StateInfo(NamedTuple):
    """状态分类结果"""
    state: EvolutionaryState
    evolution_factor: float      # 进化因子 f ∈ [0, 1]
    diversity_ratio: float       # 多样性比例
    stagnation_ratio: float      # 停滞比例
    confidence: float            # 分类置信度


class EvolutionaryStateClassifier:
    """
    进化状态分类器
    
    基于进化因子 f = (d_g - d_min) / (d_max - d_min) 判定当前进化状态
    
    其中：
    - d_g: 粒子到全局最优的平均距离
    - d_min, d_max: 距离分布的最小/最大值
    
    状态边界（可调节）：
    - f > 0.7: Exploration (高多样性，充分探索)
    - 0.3 < f ≤ 0.7: Exploitation (适度收敛，局部精搜)
    - 0.1 < f ≤ 0.3: Convergence (高度收敛，精细开发)
    - f ≤ 0.1 且停滞: Jump-out (需要逃离)
    """
    
    # 状态边界阈值
    EXPLORATION_THRESHOLD = 0.7      # f > 0.7 → 探索
    EXPLOITATION_THRESHOLD = 0.3     # 0.3 < f ≤ 0.7 → 开发
    CONVERGENCE_THRESHOLD = 0.1      # 0.1 < f ≤ 0.3 → 收敛
    
    # 停滞判定阈值
    STAGNATION_PATIENCE = 5          # 连续5次无改进视为停滞
    
    # 每种状态的建议参数
    STATE_PARAMS = {
        EvolutionaryState.EXPLORATION: {
            'w': 0.9, 'c1': 1.2, 'c2': 2.0,
            'description': '全局搜索：高惯性，偏社会学习'
        },
        EvolutionaryState.EXPLOITATION: {
            'w': 0.5, 'c1': 2.0, 'c2': 1.5,
            'description': '局部精搜：中等惯性，偏认知学习'
        },
        EvolutionaryState.CONVERGENCE: {
            'w': 0.2, 'c1': 2.5, 'c2': 2.5,
            'description': '收敛开发：极低惯性，强拉力'
        },
        EvolutionaryState.JUMP_OUT: {
            'w': 0.9, 'c1': 0.5, 'c2': 0.5,
            'description': '逃离局部：高惯性，弱拉力，触发NDM'
        }
    }
    
    def __init__(self, 
                 exploration_threshold: float = 0.7,
                 exploitation_threshold: float = 0.3,
                 convergence_threshold: float = 0.1,
                 stagnation_patience: int = 5):
        """
        Args:
            exploration_threshold: 探索状态阈值
            exploitation_threshold: 开发状态阈值
            convergence_threshold: 收敛状态阈值
            stagnation_patience: 停滞耐心值
        """
        self.exploration_threshold = exploration_threshold
        self.exploitation_threshold = exploitation_threshold
        self.convergence_threshold = convergence_threshold
        self.stagnation_patience = stagnation_patience
    
    def compute_evolution_factor(self, 
                                  positions: np.ndarray,
                                  gbest_position: np.ndarray) -> Tuple[float, float, float]:
        """
        计算进化因子
        
        Args:
            positions: 粒子位置 shape (n_particles, dim)
            gbest_position: 全局最优位置 shape (dim,)
            
        Returns:
            f: 进化因子
            d_mean: 平均距离
            d_std: 距离标准差
        """
        # 计算所有粒子到全局最优的距离
        distances = np.linalg.norm(positions - gbest_position, axis=1)
        
        d_min = distances.min()
        d_max = distances.max()
        d_mean = distances.mean()
        d_std = distances.std()
        
        # 进化因子 f = (d_g - d_min) / (d_max - d_min)
        # 即平均距离在距离范围中的相对位置
        if d_max - d_min < 1e-10:
            # 所有粒子几乎在同一位置
            f = 0.0
        else:
            f = (d_mean - d_min) / (d_max - d_min + 1e-10)
        
        return f, d_mean, d_std
    
    def classify(self,
                 positions: np.ndarray,
                 gbest_position: np.ndarray,
                 stagnation_count: int,
                 prev_best_fitness: float = None,
                 current_best_fitness: float = None) -> StateInfo:
        """
        分类当前进化状态
        
        Args:
            positions: 粒子位置 shape (n_particles, dim)
            gbest_position: 全局最优位置 shape (dim,)
            stagnation_count: 停滞计数
            prev_best_fitness: 上一轮最优适应度（可选）
            current_best_fitness: 当前最优适应度（可选）
            
        Returns:
            StateInfo: 状态分类结果
        """
        # 计算进化因子
        f, d_mean, d_std = self.compute_evolution_factor(positions, gbest_position)
        
        # 计算多样性比例（标准差归一化）
        n_particles, dim = positions.shape
        # 使用坐标范围估算理论最大标准差
        pos_range = positions.max() - positions.min()
        diversity_ratio = min(1.0, d_std / (pos_range / 2 + 1e-10))
        
        # 计算停滞比例
        stagnation_ratio = min(1.0, stagnation_count / (2 * self.stagnation_patience))
        
        # 状态分类逻辑
        is_stagnating = stagnation_count >= self.stagnation_patience
        
        if f <= self.convergence_threshold and is_stagnating:
            # 高度收敛 + 停滞 → 需要跳出
            state = EvolutionaryState.JUMP_OUT
            confidence = min(1.0, stagnation_ratio + (1 - f))
        elif f > self.exploration_threshold:
            # 高进化因子 → 探索阶段
            state = EvolutionaryState.EXPLORATION
            confidence = (f - self.exploration_threshold) / (1 - self.exploration_threshold + 1e-10)
        elif f > self.exploitation_threshold:
            # 中等进化因子 → 开发阶段
            state = EvolutionaryState.EXPLOITATION
            range_size = self.exploration_threshold - self.exploitation_threshold
            mid_point = (self.exploration_threshold + self.exploitation_threshold) / 2
            confidence = 1 - abs(f - mid_point) / (range_size / 2 + 1e-10)
        elif f > self.convergence_threshold:
            # 低进化因子但未停滞 → 收敛阶段
            state = EvolutionaryState.CONVERGENCE
            range_size = self.exploitation_threshold - self.convergence_threshold
            confidence = (self.exploitation_threshold - f) / (range_size + 1e-10)
        else:
            # 极低进化因子但未达停滞阈值
            if stagnation_count >= self.stagnation_patience // 2:
                state = EvolutionaryState.JUMP_OUT
                confidence = 0.5 + 0.5 * stagnation_ratio
            else:
                state = EvolutionaryState.CONVERGENCE
                confidence = 0.8
        
        confidence = np.clip(confidence, 0.0, 1.0)
        
        return StateInfo(
            state=state,
            evolution_factor=f,
            diversity_ratio=diversity_ratio,
            stagnation_ratio=stagnation_ratio,
            confidence=confidence
        )
    
    def get_recommended_params(self, state: EvolutionaryState) -> Dict[str, float]:
        """
        获取状态对应的建议参数
        
        Args:
            state: 进化状态
            
        Returns:
            包含 w, c1, c2 的参数字典
        """
        params = self.STATE_PARAMS[state].copy()
        # 移除description字段
        params.pop('description', None)
        return params
    
    def get_state_name(self, state: EvolutionaryState) -> str:
        """获取状态名称"""
        names = {
            EvolutionaryState.EXPLORATION: "Exploration (S1)",
            EvolutionaryState.EXPLOITATION: "Exploitation (S2)",
            EvolutionaryState.CONVERGENCE: "Convergence (S3)",
            EvolutionaryState.JUMP_OUT: "Jump-out (S4)"
        }
        return names[state]
    
    def get_state_one_hot(self, state: EvolutionaryState) -> np.ndarray:
        """
        获取状态的 One-Hot 编码
        
        Args:
            state: 进化状态
            
        Returns:
            shape (4,) 的 One-Hot 向量
        """
        one_hot = np.zeros(4, dtype=np.float32)
        one_hot[int(state)] = 1.0
        return one_hot


class SubgroupStateTracker:
    """
    子种群状态追踪器
    
    为每个子种群维护进化状态历史，支持状态转换分析
    """
    
    def __init__(self, num_subgroups: int, history_length: int = 10):
        """
        Args:
            num_subgroups: 子种群数量
            history_length: 历史记录长度
        """
        self.num_subgroups = num_subgroups
        self.history_length = history_length
        self.classifier = EvolutionaryStateClassifier()
        
        # 每个子种群的状态历史
        self.state_history = [[] for _ in range(num_subgroups)]
        # 当前状态
        self.current_states = [None] * num_subgroups
    
    def update(self, 
               group_idx: int,
               positions: np.ndarray,
               gbest_position: np.ndarray,
               stagnation_count: int) -> StateInfo:
        """
        更新指定子种群的状态
        
        Args:
            group_idx: 子种群索引
            positions: 该组粒子位置
            gbest_position: 该组全局最优位置
            stagnation_count: 停滞计数
            
        Returns:
            StateInfo: 分类结果
        """
        state_info = self.classifier.classify(
            positions=positions,
            gbest_position=gbest_position,
            stagnation_count=stagnation_count
        )
        
        # 更新历史
        self.state_history[group_idx].append(state_info.state)
        if len(self.state_history[group_idx]) > self.history_length:
            self.state_history[group_idx].pop(0)
        
        self.current_states[group_idx] = state_info
        
        return state_info
    
    def get_state_distribution(self, group_idx: int) -> np.ndarray:
        """
        获取历史状态分布
        
        Args:
            group_idx: 子种群索引
            
        Returns:
            shape (4,) 的状态分布向量
        """
        history = self.state_history[group_idx]
        if not history:
            return np.zeros(4, dtype=np.float32)
        
        distribution = np.zeros(4, dtype=np.float32)
        for state in history:
            distribution[int(state)] += 1
        
        return distribution / len(history)
    
    def get_transition_matrix(self, group_idx: int) -> np.ndarray:
        """
        获取状态转换矩阵
        
        Args:
            group_idx: 子种群索引
            
        Returns:
            shape (4, 4) 的转换概率矩阵
        """
        history = self.state_history[group_idx]
        if len(history) < 2:
            return np.zeros((4, 4), dtype=np.float32)
        
        transitions = np.zeros((4, 4), dtype=np.float32)
        for i in range(len(history) - 1):
            from_state = int(history[i])
            to_state = int(history[i + 1])
            transitions[from_state, to_state] += 1
        
        # 归一化
        row_sums = transitions.sum(axis=1, keepdims=True)
        transitions = np.where(row_sums > 0, transitions / (row_sums + 1e-10), 0)
        
        return transitions
    
    def reset(self):
        """重置所有状态历史"""
        self.state_history = [[] for _ in range(self.num_subgroups)]
        self.current_states = [None] * self.num_subgroups


# ===== 工具函数 =====
def get_evolution_state_dim() -> int:
    """获取进化状态特征维度（One-Hot + 额外特征）"""
    return 4  # 仅 One-Hot 编码
