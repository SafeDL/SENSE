"""
调用carla自带的Autonomous driving agent
"""

import numpy as np

from safebench.agent.base_policy import BasePolicy
from safebench.carla_agents.navigation.behavior_agent import BehaviorAgent


class CarlaBehaviorAgent(BasePolicy):
    name = 'behavior'
    type = 'unlearnable'

    """ This is just an example for testing, which always goes straight. """
    def __init__(self, config, logger):
        self.logger = logger
        self.num_scenario = config['num_scenario']
        self.ego_action_dim = config['ego_action_dim']
        self.model_path = config['model_path']
        self.mode = 'train'
        self.continue_episode = 0
        self.route = None
        self.controller_list = []
        behavior_list = ["cautious", "normal", "aggressive"]
        self.behavior = behavior_list[1]

    def set_ego_and_route(self, ego_vehicles, info, static_obs=None):
        self.ego_vehicles = ego_vehicles
        self.controller_list = []
        for e_i in range(len(ego_vehicles)):  # e_i代表同时在执行的测试场景的序号
            # 获取在测试路线设计之初,就定义了的轨迹行驶速度
            controller = BehaviorAgent(self.ego_vehicles[e_i], behavior=self.behavior, target_route_speed=static_obs[e_i]['target_speed'])
            dest_waypoint = info[e_i]['route_waypoints'][-1]
            location = dest_waypoint.transform.location
            controller.set_destination(location) # 为每个控制器设置路线
            self.controller_list.append(controller)

    def train(self, replay_buffer):
        pass

    def set_mode(self, mode):
        self.mode = mode

    def get_action(self, obs, infos, deterministic=False):
        actions = []
        for e_i, _ in enumerate(infos):  # e_i代表同时在执行的测试场景的序号
            # select the controller that matches the scenario_id
            control = self.controller_list[e_i].run_step()
            throttle = control.throttle
            steer = control.steer
            actions.append([throttle, steer]) 
        actions = np.array(actions, dtype=np.float32)
        return actions

    def load_model(self):
        pass

    def save_model(self):
        pass
