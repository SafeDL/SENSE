"""
该脚本主要用于定义场景生成策略,将场景生成策略和对应的场景动作实现类一一对应
"""

# collect policy models from scenarios
from safebench.scenario.scenario_policy.dummy_policy import DummyPolicy
from safebench.scenario.scenario_policy.reinforce_continuous import REINFORCE

# 这里列出了可用的场景生成策略,例如知识的、硬编码、随机、强化学习、生成式模型等
SCENARIO_POLICY_LIST = {
    'standard': DummyPolicy,
    'ordinary': DummyPolicy,
    'scenic': DummyPolicy,
    'lc': REINFORCE,  # 可以调整的测试空间只是adversarial agent的初始位置
}
