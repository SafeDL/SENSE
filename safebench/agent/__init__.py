"""
将被测算法的类名和类对象一一对应起来，方便后续的调用
"""

# for planning scenario
from safebench.agent.dummy import DummyAgent
from safebench.agent.basic import CarlaBasicAgent
from safebench.agent.behavior import CarlaBehaviorAgent

# 注意这里列出了所有的被测算法类型,包括carla自带AD,感知（YOLO,faster_rcnn）和规划（SAC,DDPG,PPO,TD3）等
AGENT_POLICY_LIST = {
    'dummy': DummyAgent,
    'basic': CarlaBasicAgent,
    'behavior': CarlaBehaviorAgent,
}

