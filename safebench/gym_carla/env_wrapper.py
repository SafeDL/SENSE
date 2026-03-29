"""
1、通过VectorWrapper类初始化多个仿真环境,并管理这些环境的状态空间和动作空间
2、根据场景配置重置环境,创建场景和自车,并返回观察和信息
3、在每个时间步骤内，应用自车和场景的动作，更新环境的状态，并收集新观察、奖励、完成标志和信息
4、根据不同的模式(自车策略、场景策略、评估),调用相应的策略进行训练或评估,并记录和保存结果
5、在每个场景结束后,清理环境中的所有对象,确保仿真环境的正确关闭
6、日志记录和视频保存
"""

import gym
import numpy as np
import pygame


class VectorWrapper:
    """ 
        The interface to control a list of environments.
    """

    def __init__(self, env_params, scenario_config, world, birdeye_render, display, logger):
        self.logger = logger
        self.world = world
        self.num_scenario = scenario_config['num_scenario']  # 同时执行的仿真场景数量
        self.ROOT_DIR = scenario_config['ROOT_DIR']
        self.frame_skip = scenario_config['frame_skip']  
        self.render = scenario_config['render']

        self.env_list = []  # 为了实现多个场景的并行测试,需要对应创建多个gym_carla环境
        self.action_space_list = []
        for _ in range(self.num_scenario):
            env = carla_env(env_params, birdeye_render=birdeye_render, display=display, world=world, logger=logger)
            self.env_list.append(env)
            self.action_space_list.append(env.action_space)

        # flags for env list 
        self.finished_env = [False] * self.num_scenario
        self.running_results = {}
    
    def obs_postprocess(self, obs_list):
        # assume all variables are array
        obs_list = np.array(obs_list)
        return obs_list

    def get_ego_vehicles(self):
        ego_vehicles = []
        for env in self.env_list:
            if env.ego_vehicle is not None:
                # self.logger.log('>> Ego vehicle is None. Please call reset() first.', 'red')
                # raise Exception()
                ego_vehicles.append(env.ego_vehicle)
        return ego_vehicles

    def get_static_obs(self, scenario_configs):
        static_obs_list = []
        for s_i in range(len(scenario_configs)):
            static_obs = self.env_list[s_i].get_static_obs(scenario_configs[s_i])
            static_obs_list.append(static_obs)
        return static_obs_list

    def reset(self, scenario_configs, scenario_init_action):
        # create scenarios and ego vehicles
        obs_list = []
        info_list = []
        for s_i in range(len(scenario_configs)):  # 在gym_carla环境初始化的时候创建场景
            config = scenario_configs[s_i]
            obs, info = self.env_list[s_i].reset(config=config, env_id=s_i, scenario_init_action=scenario_init_action[s_i])
            obs_list.append(obs)
            info_list.append(info)
            # 标记当前执行场景的索引data_id
            info_list[s_i].update({'data_id': config.data_id})

        # sometimes not all scenarios are used
        self.finished_env = [False] * self.num_scenario
        for s_i in range(len(scenario_configs), self.num_scenario):
            self.finished_env[s_i] = True

        # store scenario id
        for s_i in range(len(scenario_configs)):
            info_list[s_i].update({'s_id': s_i})

        # return obs
        return self.obs_postprocess(obs_list), info_list

    def step(self, ego_actions, scenario_actions):
        """
            ego_actions: [num_alive_scenario]
            scenario_actions: [num_alive_scenario]
        """
        # apply action
        action_idx = 0  # action idx should match the env that is not finished
        for e_i in range(self.num_scenario):
            if not self.finished_env[e_i]:
                processed_action = self.env_list[e_i]._postprocess_action(ego_actions[action_idx])
                # TODO: pre-process scenario action
                current_env = self.env_list[e_i]
                current_env.step_before_tick(processed_action, scenario_actions[action_idx])
                action_idx += 1
        
        # tick all scenarios
        for _ in range(self.frame_skip):
            self.world.tick()

        # collect new observation of one frame
        obs_list = []
        reward_list = []
        done_list = []
        info_list = []
        for e_i in range(self.num_scenario):
            if not self.finished_env[e_i]:
                current_env = self.env_list[e_i]
                obs, reward, done, info = current_env.step_after_tick()

                # 标记当前执行的是scenario config中的第几个测试场景,注意在训练或者测试的时候,我们希望每一个测试文件的data_id都是从0开始
                info['data_id'] = current_env.config.data_id
                # store scenario id to help agent decide which policy should be used
                info['s_id'] = e_i # e_i说明的是同一个批次的第几个场景

                # check if env is done
                if done:
                    self.finished_env[e_i] = True
                    # save running results according to the data_id of scenario
                    if current_env.config.data_id in self.running_results.keys():
                        self.logger.log('Scenario with data_id {} is duplicated'.format(current_env.config.data_id))
                    self.running_results[current_env.config.data_id] = current_env.scenario_manager.running_record

                # update infomation
                obs_list.append(obs)
                reward_list.append(reward)
                done_list.append(done)
                info_list.append(info)
        
        # convert to numpy
        rewards = np.array(reward_list)
        dones = np.array(done_list)
        infos = np.array(info_list)

        # update pygame window
        if self.render:
            pygame.display.flip()
        return self.obs_postprocess(obs_list), rewards, dones, infos

    # def sample_action_space(self):
    #     action = []
    #     for action_space in self.action_space_list:
    #         action.append(action_space.sample())
    #     return np.array(action)

    def all_scenario_done(self):
        if np.sum(self.finished_env) == self.num_scenario:
            return True
        else:
            return False

    def clean_up(self):
        # 清空self.running_results
        self.running_results = {}
        # stop sensor objects
        for e_i in range(self.num_scenario):
            self.env_list[e_i].clean_up()

        # tick to ensure that all destroy commands are executed
        self.world.tick()


class ObservationWrapper(gym.Wrapper):
    def __init__(self, env, obs_type):
        # self._env是由gym_carla包创建的环境
        super().__init__(env)
        self._env = env

        self.is_running = False
        self.obs_type = obs_type
        self._build_obs_space()

        # build action space, assume the obs range from -1 to 1
        act_dim = 2
        act_lim = np.ones((act_dim), dtype=np.float32)
        # 创建一个Box类型的动作空间,动作空间的每个维度的取值都是从-act_lim到act_lim,数据类型为np.float32
        self.action_space = gym.spaces.Box(-act_lim, act_lim, dtype=np.float32)

    def get_static_obs(self, config):
        return self._env.get_static_obs(config)

    def reset(self, **kwargs):
        # 使用gym_carla包重置环境并返回观察和信息
        # 其中obs是一个字典,包含了环境的初始状态信息,传感器数据（相机、激光雷达）和车辆状态（与车道线的相对位置、速度等）
        # info也是一个字典,包含route_waypoints,cost和actor_info等信息
        obs, info = self._env.reset(**kwargs)
        return self._preprocess_obs(obs), info

    def step_before_tick(self, ego_action, scenario_action):
        self._env.step_before_tick(ego_action=ego_action, scenario_action=scenario_action)

    def step_after_tick(self):
        obs, reward, done, info = self._env.step_after_tick()
        self.is_running = self._env.is_running
        reward, info = self._preprocess_reward(reward, info)
        obs = self._preprocess_obs(obs)
        return obs, reward, done, info

    def _build_obs_space(self):
        if self.obs_type == 0:
            obs_dim = 4
            # assume the obs range from -1 to 1
            obs_lim = np.ones((obs_dim), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-obs_lim, obs_lim)
        elif self.obs_type == 1:
            obs_dim = 11
            # assume the obs range from -1 to 1
            obs_lim = np.ones((obs_dim), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-obs_lim, obs_lim)
        elif self.obs_type == 2 or self.obs_type == 3:
            # 4 state space + bev
            obs_dim = 128  # TODO: should be the same as display_size
            # assume the obs range from -1 to 1
            obs_lim = np.ones((obs_dim), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-obs_lim, obs_lim)
        else:
            raise NotImplementedError

    def _preprocess_obs(self, obs):
        """
        根据不同的obs_type,对环境数据进行预处理,具体操作为
            1、obs_type=0: 只使用4维状态空间
            2、obs_type=1: 连接4维状态空间和车道信息
            3、obs_type=2: 返回一个包含鸟瞰图和状态的字典
            4、obs_type=3: 返回一个包含前视图和状态的字典
            其他情况: 抛出NotImplementedError异常
        """

        # only use the 4-dimensional state space
        if self.obs_type == 0:
            return obs['state'][:4].astype(np.float64)
        # concat the 4-dimensional state space and lane info
        elif self.obs_type == 1:
            new_obs = np.array([
                obs['state'][0], obs['state'][1], obs['state'][2], obs['state'][3],
                obs['command'], 
                obs['forward_vector'][0], obs['forward_vector'][1],
                obs['node_forward'][0], obs['node_forward'][1],
                obs['target_forward'][0], obs['target_forward'][1]
            ])
            return new_obs
        # return a dictionary with bird-eye view image and state
        elif self.obs_type == 2:
            return {"img": obs['birdeye'], "states": obs['state'][:4].astype(np.float64)}
        # return a dictionary with front-view image and state
        elif self.obs_type == 3:
            return {"img": obs['camera'], "states": obs['state'][:4].astype(np.float64)}
        else:
            raise NotImplementedError

    def _preprocess_reward(self, reward, info):
        return reward, info

    def _postprocess_action(self, action):
        return action

    def clear_up(self):
        self._env.clear_up()


def carla_env(env_params, birdeye_render=None, display=None, world=None, logger=None):
    # 查找注册为'carla-v0'的环境,并且实例化它
    return ObservationWrapper(
        gym.make(
            'carla-v0', 
            env_params=env_params, 
            birdeye_render=birdeye_render,
            display=display, 
            world=world, 
            logger=logger,
        ), 
        obs_type=env_params['obs_type']
    )
