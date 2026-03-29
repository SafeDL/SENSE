"""
base_env.py - 优化环境抽象基类
定义统一的环境接口，便于在数学基准和自动驾驶场景间切换
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np


class BaseOptEnv(ABC):
    """
    优化环境抽象基类
    
    所有环境必须实现:
    - evaluate(): 评估粒子位置,返回适应度和不确定性
    - get_bounds(): 返回搜索空间边界
    - get_dim(): 返回搜索空间维度
    """
    
    def __init__(self, dim: int, bounds: Tuple[float, float]):
        """
        Args:
            dim: 搜索空间维度
            bounds: 搜索空间边界 (lower, upper)
        """
        self.dim = dim
        self.bounds = bounds
        self.evaluation_count = 0
    
    @abstractmethod
    def evaluate(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        评估粒子位置
        
        Args:
            positions: shape (N, dim) 的粒子位置矩阵
            
        Returns:
            fitness: shape (N,) 的适应度值（越小越好）
            uncertainty: shape (N,) 的不确定性值（数学函数为 0）
        """
        pass
    
    def get_bounds(self) -> Tuple[float, float]:
        """
        返回搜索空间边界
        
        Returns:
            bounds: (lower, upper) 元组
        """
        return self.bounds
    
    def get_dim(self) -> int:
        """
        返回搜索空间维度
        
        Returns:
            dim: 维度数
        """
        return self.dim
    
    def get_evaluation_count(self) -> int:
        """返回累计评估次数"""
        return self.evaluation_count
    
    def reset_evaluation_count(self):
        """重置评估计数器"""
        self.evaluation_count = 0
    
    def random_positions(self, n: int) -> np.ndarray:
        """
        生成 n 个随机位置
        
        Args:
            n: 粒子数量
            
        Returns:
            positions: shape (n, dim) 的随机位置
        """
        lower, upper = self.bounds
        return lower + (upper - lower) * np.random.rand(n, self.dim)
    
    def clip_positions(self, positions: np.ndarray) -> np.ndarray:
        """
        将位置裁剪到边界内
        
        Args:
            positions: 粒子位置
            
        Returns:
            clipped: 裁剪后的位置
        """
        lower, upper = self.bounds
        return np.clip(positions, lower, upper)
    
    def get_global_optimum(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        返回全局最优解（如果已知）
        
        Returns:
            (position, fitness) 或 None
        """
        return None
    
    def get_info(self) -> dict:
        """返回环境信息"""
        return {
            'dim': self.dim,
            'bounds': self.bounds,
            'evaluation_count': self.evaluation_count,
        }
