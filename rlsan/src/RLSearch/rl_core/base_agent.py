"""
base_agent.py - 统一的RL Agent基类接口

为所有RL算法（DDQN、离散PPO、连续PPO等）定义统一接口，
便于在StateAwareNichePSO中无缝切换不同的RL算法。

设计原则：
- 统一的 choose_action / store_transition / update 接口
- 通过 action_type 属性区分离散/连续动作
- 支持训练模式和部署模式
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict, Optional
import numpy as np


class BaseRLAgent(ABC):
    """
    所有RL Agent的抽象基类
    
    子类需要实现：
    - choose_action: 根据状态选择动作
    - store_transition: 存储经验
    - update: 更新策略
    
    属性：
    - action_type: 'discrete' 或 'continuous'
    - frozen: 是否处于部署模式（不更新）
    """
    
    def __init__(self):
        self._frozen = False
    
    @property
    @abstractmethod
    def action_type(self) -> str:
        """返回动作类型: 'discrete' 或 'continuous'"""
        pass
    
    @property
    def frozen(self) -> bool:
        """是否处于冻结状态（部署模式）"""
        return self._frozen
    
    @frozen.setter
    def frozen(self, value: bool):
        self._frozen = value
    
    @abstractmethod
    def choose_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[Any, Any]:
        """
        根据状态选择动作
        
        Args:
            state: 当前状态向量
            deterministic: 是否使用确定性策略（无探索）
            
        Returns:
            action: 动作（离散为int，连续为np.ndarray）
            action_info: 附加信息（离散为概率分布，连续为log_prob等）
        """
        pass
    
    @abstractmethod
    def store_transition(self, state: np.ndarray, action: Any, 
                        reward: float, next_state: np.ndarray, 
                        done: bool = False):
        """
        存储一条经验
        
        Args:
            state: 当前状态
            action: 采取的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否终止
        """
        pass
    
    @abstractmethod
    def update(self) -> Dict[str, float]:
        """
        更新策略
        
        Returns:
            metrics: 训练指标字典（loss等）
        """
        pass
    
    def freeze(self):
        """冻结Agent（进入部署模式）"""
        self._frozen = True
    
    def unfreeze(self):
        """解冻Agent（进入训练模式）"""
        self._frozen = False
    
    @abstractmethod
    def save(self, path: str):
        """保存模型权重"""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str, freeze: bool = True) -> 'BaseRLAgent':
        """从文件加载模型"""
        pass


class DiscreteActionMixin:
    """离散动作Agent的Mixin类"""
    
    @property
    def action_type(self) -> str:
        return 'discrete'
    
    @property
    def num_actions(self) -> int:
        """返回离散动作数量"""
        raise NotImplementedError


class ContinuousActionMixin:
    """连续动作Agent的Mixin类"""
    
    @property
    def action_type(self) -> str:
        return 'continuous'
    
    @property
    def action_dim(self) -> int:
        """返回连续动作维度"""
        raise NotImplementedError
    
    @property
    def action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回动作空间的上下界 (low, high)"""
        raise NotImplementedError


# 动作参数范围常量（供连续动作Agent使用）
PSO_ACTION_BOUNDS = {
    'w': (0.1, 0.9),           # 惯性权重
    'c1': (0.5, 3.0),          # 认知系数
    'c2': (0.5, 3.0),          # 社会系数
    'velocity_scale': (0.1, 2.0),  # 速度缩放因子
}

# 默认动作参数（用于初始化）
PSO_ACTION_DEFAULTS = {
    'w': 0.729,
    'c1': 1.494,
    'c2': 1.494,
    'velocity_scale': 1.0,
}
