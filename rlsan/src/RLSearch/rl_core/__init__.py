"""
rl_core - RL 智能体核心模块

包含:
- base_agent: 抽象基类和 Mixin
- networks: MLP, EnhancedDQN 等网络架构
- buffers: ReplayBuffer, OUNoise 等共享组件
- 各算法 Agent: DQN, DDPG, SAC, TD3, PPO, TRPO, Random
"""

try:
    from .networks import MLP, EnhancedDQN, RunningMeanStd
    from .base_agent import BaseRLAgent, DiscreteActionMixin, ContinuousActionMixin
    from .buffers import ReplayBuffer, OUNoise
    from .dqn_agent import DoubleDQNAgent
    from .ddpg_agent import DDPGAgent
    from .sac_agent import SACAgent
    from .td3_agent import TD3Agent
    from .ppo_agent import PPOAgent
    from .trpo_agent import TRPOAgent
    from .random_agent import RandomAgent
except ImportError:
    from networks import MLP, EnhancedDQN, RunningMeanStd
    from base_agent import BaseRLAgent, DiscreteActionMixin, ContinuousActionMixin
    from buffers import ReplayBuffer, OUNoise
    from dqn_agent import DoubleDQNAgent
    from ddpg_agent import DDPGAgent
    from sac_agent import SACAgent
    from td3_agent import TD3Agent
    from ppo_agent import PPOAgent
    from trpo_agent import TRPOAgent
    from random_agent import RandomAgent

__all__ = [
    # 基础
    'BaseRLAgent', 'DiscreteActionMixin', 'ContinuousActionMixin',
    # 网络
    'MLP', 'EnhancedDQN', 'RunningMeanStd',
    # 共享组件
    'ReplayBuffer', 'OUNoise',
    # Agent
    'DoubleDQNAgent', 'DDPGAgent', 'SACAgent', 'TD3Agent',
    'PPOAgent', 'TRPOAgent', 'RandomAgent',
]
