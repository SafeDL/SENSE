"""
action_space.py - 状态感知动作空间 (增强版)

核心改进：
1. 集成进化状态分类器，自动推荐最佳策略
2. Agent只需决定是否遵循推荐，大幅简化学习任务

增强版改进：
- 从"静态调参"转向"动态算子" - 动作定义为搜索模式
- 小生境动态半径 - RL实时输出半径调整系数
- 速度调整 - 支持激励/抑制模式
- 邻域差分变异 (NDM) - 停滞时强行改变演化轨迹

动作空间设计（8个行为模式）：
- Action 0: 广域侦察 (高w, 大半径, 低社会学习)
- Action 1: 局部围攻 (低w, 小半径, 高认知学习)
- Action 2: 平衡搜索 (中等w, 标准参数)
- Action 3: 小生境逃逸 (触发NDM + 重置部分粒子)
- Action 4: 半径扩张 (半径×1.5, 增加覆盖)
- Action 5: 半径收缩 (半径×0.7, 精细搜索)
- Action 6: 速度激励 (速度×1.3, 加速演化)
- Action 7: 速度抑制 (速度×0.5, 减缓收敛)
"""

import numpy as np
from typing import Dict, Tuple, Optional

# 本地导入
from .evo_state import (
    EvolutionaryState, EvolutionaryStateClassifier, StateInfo
)



class StateAwareActionSpace:
    """
    状态感知动作空间 (增强版)
    
    设计理念：
    - 动作不再仅是 w, c1, c2 的数值，而是定义为搜索模式
    - 每个动作对应明确的行为逻辑（广域侦察、局部围攻、小生境逃逸等）
    - 支持速度调整
    - 移除动态半径调整（niche_radius 固定）
    """
    
    BEHAVIOR_MODES = {
        # Action 0: 广域侦察 - 高w, 低社会学习
        0: {
            'name': 'wide_scout',
            'w': 0.9, 'c1': 0.8, 'c2': 2.5, # c1更低，c2更高 -> 偏向全局
            'velocity_scale': 1.0,
            'trigger_ndm': False,
            'description': '广域侦察: 高惯性探索新区域'
        },
        # Action 1: 局部围攻 - 低w, 高认知学习
        1: {
            'name': 'local_attack',
            'w': 0.1, 'c1': 3.0, 'c2': 0.5, # w极低，c1极高 -> 强认知
            'velocity_scale': 0.5,
            'trigger_ndm': False,
            'description': '局部围攻: 低惯性精细搜索'
        },
        # Action 2: 平衡搜索 - 中等w, 标准参数
        2: {
            'name': 'balanced',
            'w': 0.6, 'c1': 1.49, 'c2': 1.49, # 经典参数
            'velocity_scale': 1.0,
            'trigger_ndm': False,
            'description': '平衡搜索: 标准参数配置'
        },
        # Action 3: 小生境逃逸 - 触发NDM
        3: {
            'name': 'niche_escape',
            'w': 0.729, 'c1': 1.49, 'c2': 1.49,
            'velocity_scale': 1.5,
            'trigger_ndm': True,  # 触发邻域差分变异
            'description': '小生境逃逸: NDM强制改变轨迹'
        },
        # Action 4: 速度激励 - 加速演化
        4: {
            'name': 'velocity_boost',
            'w': 0.8, 'c1': 1.5, 'c2': 1.5,
            'velocity_scale': 2.0,  # 速度极大放大
            'trigger_ndm': False,
            'description': '速度激励: 加速粒子运动'
        },
        # Action 5: 速度抑制 - 减缓收敛
        5: {
            'name': 'velocity_dampen',
            'w': 0.4, 'c1': 1.8, 'c2': 1.8,
            'velocity_scale': 0.1,  # 速度极小 (冻结)
            'trigger_ndm': False,
            'description': '速度抑制: 减缓运动速度'
        },
    }
    
    def __init__(self):
        """初始化状态感知动作空间 (增强版)"""
        self.num_actions = len(self.BEHAVIOR_MODES)
        self.classifier = EvolutionaryStateClassifier()
        
        # 动作类型映射
        self.action_type_map = {
            i: mode['name'] for i, mode in self.BEHAVIOR_MODES.items()
        }
    
    def get_num_actions(self) -> int:
        """返回动作空间大小"""
        return self.num_actions
    
    def get_recommended_action(self, state_info: StateInfo) -> int:
        """
        根据进化状态获取推荐动作
        
        Args:
            state_info: 进化状态分类结果
            
        Returns:
            recommended_action: 推荐的动作索引
        """
        # 根据进化状态推荐匹配的行为模式
        state_to_action = {
            EvolutionaryState.EXPLORATION: 0,   # 广域侦察
            EvolutionaryState.EXPLOITATION: 1,  # 局部围攻
            EvolutionaryState.CONVERGENCE: 5,   # 速度抑制 (替代半径收缩)
            EvolutionaryState.JUMP_OUT: 3,      # 小生境逃逸
        }
        return state_to_action.get(state_info.state, 2)  # 默认平衡
    
    def get_action_params(self, 
                          action_idx: int,
                          state_info: Optional[StateInfo] = None,
                          stagnation: int = 0,
                          diversity: float = 0.5) -> Dict:
        """
        获取动作对应的PSO参数 (增强版)
        
        Args:
            action_idx: 动作索引 [0, 5]
            state_info: 进化状态分类结果
            stagnation: 停滞计数
            diversity: 多样性
            
        Returns:
            params: 包含 w, c1, c2, velocity_scale, trigger_ndm 的字典
        """
        action_idx = action_idx % self.num_actions
        mode = self.BEHAVIOR_MODES[action_idx]
        
        params = {
            'w': mode['w'],
            'c1': mode['c1'],
            'c2': mode['c2'],
            'velocity_scale': mode.get('velocity_scale', 1.0),
            'trigger_ndm': mode.get('trigger_ndm', False)
        }
        
        # 严重停滞时强制触发NDM
        if stagnation > 8 and not params['trigger_ndm']:
            params['trigger_ndm'] = True
        
        return params
    
    def get_action_name(self, action_idx: int) -> str:
        """获取动作的可读名称"""
        return self.action_type_map.get(action_idx % self.num_actions, f"action_{action_idx}")
    
    def compute_action_reward_modifier(self, 
                                        action_idx: int,
                                        state_info: StateInfo) -> float:
        """
        计算动作匹配奖励修正
        """
        # 简单实现：如果是推荐动作则奖励
        rec_action = self.get_recommended_action(state_info)
        if action_idx == rec_action:
            return 0.1 * state_info.confidence
        return 0.0
    
    def print_action_summary(self):
        """打印动作空间摘要"""
        print("\n" + "=" * 60)
        print("StateAwareActionSpace - Action Summary (6 Actions)")
        print("=" * 60)
        
        for i, mode in self.BEHAVIOR_MODES.items():
            print(f"  Action {i}: {mode['name']}")
            print(f"    {mode['description']}")
            print(f"    w={mode['w']:.2f}, c1={mode['c1']:.2f}, c2={mode['c2']:.2f}")
        
        print("=" * 60 + "\n")


# ===== 工具函数 =====

def get_state_aware_action_dim() -> int:
    """获取状态感知动作空间大小"""
    return 6  # 固定为6个动作


# ===== 测试代码 =====
if __name__ == "__main__":
    # 创建动作空间
    action_space = StateAwareActionSpace()
    action_space.print_action_summary()
    
    # 模拟一些状态
    from evo_state import StateInfo
    
    print("Testing with different evolutionary states:")
    print("-" * 50)
    
    test_states = [
        StateInfo(EvolutionaryState.EXPLORATION, 0.8, 0.6, 0.1, 0.9),
        StateInfo(EvolutionaryState.EXPLOITATION, 0.5, 0.4, 0.2, 0.85),
        StateInfo(EvolutionaryState.CONVERGENCE, 0.2, 0.2, 0.3, 0.7),
        StateInfo(EvolutionaryState.JUMP_OUT, 0.05, 0.1, 0.8, 0.95),
    ]
    
    for state_info in test_states:
        print(f"\n📊 State: {state_info.state.name}")
        rec = action_space.get_recommended_action(state_info)
        print(f"   Recommended: {rec} ({action_space.get_action_name(rec)})")

