"""
random_agent.py - 随机策略Agent

用于对比实验，遵循与DoubleDQNAgent相同的接口。
随机选择动作，不进行任何学习。
"""

import numpy as np
from typing import Tuple, Optional


class RandomAgent:
    """
    随机策略Agent
    
    在每个决策步均匀随机选择动作，用于与RL策略进行对比实验。
    
    接口与DoubleDQNAgent保持一致，便于无缝替换。
    
    Args:
        state_dim: 状态维度（仅用于接口兼容，不实际使用）
        num_actions: 动作空间大小
        seed: 随机种子（可选）
    """
    
    def __init__(self, 
                 state_dim: int = 25,
                 num_actions: int = 6,
                 seed: Optional[int] = None):
        self.state_dim = state_dim
        self.num_actions = num_actions
        
        # 随机数生成器
        self.rng = np.random.default_rng(seed)
        
        # 始终处于"冻结"状态（不进行训练）
        self.frozen = True
        
        # 统计信息
        self.action_counts = np.zeros(num_actions, dtype=np.int32)
        self.total_steps = 0
    
    def choose_action(self, state: np.ndarray, deterministic: bool = False, temperature: float = 1.0) -> Tuple[int, np.ndarray]:
        """
        随机选择动作
        
        Args:
            state: 当前状态（不使用，仅接口兼容）
            deterministic: 是否确定性选择（忽略，始终随机）
            temperature: 温度系数（忽略）
            
        Returns:
            action: 随机选择的动作索引
            probs: 均匀概率分布
        """
        # 均匀随机选择动作
        action = self.rng.integers(0, self.num_actions)
        
        # 均匀概率分布
        probs = np.ones(self.num_actions, dtype=np.float32) / self.num_actions
        
        # 更新统计
        self.action_counts[action] += 1
        self.total_steps += 1
        
        return action, probs
    
    def store_transition(self, state: np.ndarray, action: int,
                        reward: float, next_state: np.ndarray):
        """
        存储转移（空操作，不进行学习）
        """
        pass
    
    def update_q_values(self) -> Tuple[Optional[float], Optional[float]]:
        """
        更新Q值（空操作）
        
        Returns:
            (None, None): 不进行更新
        """
        return None, None
    
    def soft_update_target_network(self, tau: float = 0.005):
        """软更新目标网络（空操作）"""
        pass
    
    def decay_epsilon(self):
        """衰减探索率（空操作）"""
        pass
    
    def freeze(self):
        """冻结Agent（已始终冻结）"""
        self.frozen = True
    
    def unfreeze(self):
        """解冻Agent（忽略，保持冻结）"""
        pass  # 随机Agent始终保持冻结
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'type': 'RandomAgent',
            'num_actions': self.num_actions,
            'total_steps': self.total_steps,
            'action_distribution': self.action_counts.tolist(),
            'action_frequencies': (self.action_counts / max(1, self.total_steps)).tolist(),
            'frozen': self.frozen,
        }
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        获取Q值（返回均匀值，用于可视化兼容）
        
        Args:
            state: 当前状态
            
        Returns:
            q_values: 全为0的数组（随机策略无Q值概念）
        """
        return np.zeros(self.num_actions, dtype=np.float32)
    
    @classmethod
    def load(cls, path: str = None, freeze: bool = True) -> 'RandomAgent':
        """
        加载Agent（创建新实例，忽略路径）
        
        用于接口兼容，实际不加载任何权重。
        
        Args:
            path: 权重路径（忽略）
            freeze: 是否冻结（忽略，始终冻结）
            
        Returns:
            agent: 新的RandomAgent实例
        """
        print(f"[RandomAgent] Created (path={path} ignored, using random policy)")
        return cls()
    
    def save(self, path: str):
        """保存Agent（空操作，无需保存）"""
        print(f"[RandomAgent] No weights to save (path={path})")
    
    def __repr__(self):
        return f"RandomAgent(num_actions={self.num_actions}, steps={self.total_steps})"


# ===== 测试代码 =====
if __name__ == "__main__":
    print("Testing RandomAgent...")
    
    agent = RandomAgent(state_dim=25, num_actions=6, seed=42)
    
    # 模拟100次动作选择
    dummy_state = np.random.randn(25)
    
    for _ in range(100):
        action, probs = agent.choose_action(dummy_state)
    
    print(f"Agent: {agent}")
    print(f"Stats: {agent.get_stats()}")
    
    # 测试接口兼容性
    agent.store_transition(dummy_state, 0, 1.0, dummy_state)
    agent.update_q_values()
    agent.decay_epsilon()
    
    print("All interface tests passed!")
