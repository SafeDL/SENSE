"""
一个用于测试的简单的Agent,继承自BasePolicy
"""

import numpy as np

from safebench.agent.base_policy import BasePolicy


class DummyAgent(BasePolicy):
    name = 'dummy'
    type = 'unlearnable'

    """ This is just an example for testing, whcih always goes straight. """
    def __init__(self, config, logger):
        self.logger = logger
        self.ego_action_dim = config['ego_action_dim']
        self.model_path = config['model_path']
        self.mode = 'train'
        self.continue_episode = 0

    def train(self, replay_buffer):
        pass

    def set_mode(self, mode):
        self.mode = mode

    def get_action(self, obs, infos, deterministic=False):
        # 输入应形成一个批次，返回的动作也应是一个批次
        batch_size = len(obs)
        action = np.random.randn(batch_size, self.ego_action_dim)
        action[:, 0] = 0.2
        action[:, 1] = 0
        return action

    def load_model(self):
        pass

    def save_model(self):
        pass
