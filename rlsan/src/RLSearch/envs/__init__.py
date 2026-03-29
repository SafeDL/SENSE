"""
Environments Module - 优化环境封装

文件结构:
- base_env.py: 环境基类
- ad_scenario_env.py: 自动驾驶场景环境
"""

try:
    from .base_env import BaseOptEnv
    from .ad_scenario_env import ADScenarioEnv
except (ImportError, ValueError):
    from base_env import BaseOptEnv
    from ad_scenario_env import ADScenarioEnv

__all__ = [
    'BaseOptEnv',
    'ADScenarioEnv'
]
