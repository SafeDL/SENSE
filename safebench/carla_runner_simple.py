"""
配合scripts/run_rlsan_search.py,读取测试参数,调用CARLA simulator 运行场景、返回测试结果
"""
import numpy as np
import carla
import pygame
from rlsan.src.sampling.utils import get_test_result, screen_most_risky_scores
import os
from safebench.gym_carla.env_wrapper import VectorWrapper
from safebench.gym_carla.envs.render import BirdeyeRender
from safebench.agent import AGENT_POLICY_LIST
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_data_loader import ScenarioDataLoaderSimple
from safebench.scenario.tools.scenario_utils import scenario_parse_simple
from safebench.util.logger import Logger, setup_logger_kwargs
from safebench.util.metric_util import get_route_scores_simple


class CarlaRunner:
    def __init__(self, agent_config, scenario_config, step_by_step=False):
        # 这里的step_by_step是为了在每次运行的时候,都能将上次的测试结果清空
        self.step_by_step = step_by_step
        self.scenario_config = scenario_config
        self.agent_config = agent_config

        self.seed = agent_config['seed']
        self.exp_name = agent_config['exp_name']
        self.output_dir = agent_config['output_dir']
        self.save_video = agent_config['save_video']

        self.render = agent_config['render']
        self.num_scenario = agent_config['num_scenario']
        self.fixed_delta_seconds = agent_config['fixed_delta_seconds']
        self.scenario_category = scenario_config['scenario_category']

        if self.step_by_step:
            # 确保每次运行的时候文件夹被清空(用于代理模型的迭代步进更新)
            exp_dir = os.path.join('/home/hp/SENSE/log/baselines', self.exp_name)
            if os.path.exists(exp_dir):
                os.system('rm -rf ' + exp_dir)
        else:
            pass

        # apply settings to carla
        self.client = carla.Client('localhost', agent_config['port'])
        self.client.set_timeout(60.0)
        self.world = None
        self.env = None

        # env_params 主要用于渲染的BirdEyeView展示
        self.env_params = {
            'auto_ego': agent_config['auto_ego'],
            'obs_type': agent_config['obs_type'],
            'scenario_category': self.scenario_category,               # planning or perception
            'ROOT_DIR': agent_config['ROOT_DIR'],
            'warm_up_steps': 9,                                        # number of ticks after spawning the vehicles
            'disable_lidar': True,                                     # show bird-eye view lidar or not
            'display_size': 256,                                       # screen size of one bird-eye view window
            'obs_range': 64,                                           # observation range (meter)
            'd_behind': 12,                                            # distance behind the ego vehicle (meter)
            'max_past_step': 1,                                        # the number of past steps to draw
            'discrete': False,                                         # whether to use discrete control space
            'discrete_acc': [-3.0, 0.0, 3.0],                          # discrete value of accelerations
            'discrete_steer': [-0.2, 0.0, 0.2],                        # discrete value of steering angles
            'continuous_accel_range': [-3.0, 3.0],                     # continuous acceleration range
            'continuous_steer_range': [-0.3, 0.3],                     # continuous steering angle range
            'max_episode_step': scenario_config['max_episode_step'],   # maximum timesteps per episode
            'max_waypt': 12,                                           # maximum number of waypoints,在BEV渲染图上绘制的路径
            'lidar_bin': 0.125,                                        # bin size of lidar sensor (meter)
            'out_lane_thres': 4,                                       # threshold for out of lane (meter)
            'desired_speed': 25,                                       # 预设的测试路线上的行驶速度 (km/h)
            'image_sz': 1024,                                          # front camera image size
        }

        # pass config from scenario to agent
        agent_config['ego_action_dim'] = scenario_config['ego_action_dim']
        agent_config['ego_state_dim'] = scenario_config['ego_state_dim']
        agent_config['ego_action_limit'] = scenario_config['ego_action_limit']

        # 定义日志记录器
        logger_kwargs = setup_logger_kwargs(
            self.exp_name,
            self.output_dir,
            self.seed,
            agent=agent_config['policy_type'],
            scenario=agent_config['policy_type'],
            scenario_category=self.scenario_category
        )
        self.logger = Logger(**logger_kwargs)  # 对Kwargs进行解包成字典
        
        # prepare parameters
        self.save_freq = scenario_config['save_freq']  # 指定数据的保存频率
        self.logger.log('>> Evaluation Mode, skip config saving', 'yellow')
        self.logger.create_eval_dir(load_existing_results=True)

        # define agent and scenario
        self.logger.log('>> Agent Policy: ' + agent_config['policy_type'])

        if self.agent_config['auto_ego']:
            self.logger.log('>> Using auto-pilot for ego vehicle, action of policy will be ignored', 'yellow')
        self.logger.log('>> ' + '-' * 40)

        # 初始化ego vehicle agent和scenario agent的策略
        self.agent_policy = AGENT_POLICY_LIST[agent_config['policy_type']](agent_config, logger=self.logger)
        if self.save_video:
            self.logger.init_video_recorder()

        # 考虑到我们的研究中map每次测试都是固定的,可以挪动到这里避免终端反复报错
        config_by_map = scenario_parse_simple(self.scenario_config, self.logger)
        for map in config_by_map.keys():  # map是地图的名称
            # initialize map and render
            self._init_world(map)
            self._init_renderer()

    def _init_world(self, town):
        self.logger.log(f">> Initializing carla world: {town}")
        self.world = self.client.load_world(town)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.fixed_delta_seconds
        self.world.apply_settings(settings)
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(self.scenario_config['tm_port'])
        # 初始化的时候设置天气为晴天,这里可以根据需求配置为设定的其他天气种类
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

    def _init_renderer(self):
        self.logger.log(">> Initializing pygame birdeye renderer")
        pygame.init()
        # 设置'pygame'显示窗口的标志位，HWSURFACE表示使用硬件加速，DOUBLEBUF表示双缓冲
        flag = pygame.HWSURFACE | pygame.DOUBLEBUF
        if not self.render:
            flag = flag | pygame.HIDDEN
        # 规划测试的显示窗口设置
        if self.scenario_category == 'planning': 
            # [bird-eye view, Lidar, front view] or [bird-eye view, front view]
            if self.env_params['disable_lidar']:
                window_size = (self.env_params['display_size'] * 2, self.env_params['display_size'] * self.num_scenario)
            else:
                window_size = (self.env_params['display_size'] * 3, self.env_params['display_size'] * self.num_scenario)
        # 感知测试的显示窗口设置
        else:
            window_size = (self.env_params['display_size'], self.env_params['display_size'] * self.num_scenario)
        # flag设置窗口属性,以下的self.display是一个可显示的窗口(512*256)
        self.display = pygame.display.set_mode(window_size, flag)

        # initialize the render for generating observation and visualization
        pixels_per_meter = self.env_params['display_size'] / self.env_params['obs_range']  # 计算观察范围内每米对应的像素数
        pixels_ahead_vehicle = (self.env_params['obs_range'] / 2 - self.env_params['d_behind']) * pixels_per_meter
        self.birdeye_params = {
            'screen_size': [self.env_params['display_size'], self.env_params['display_size']],
            'pixels_per_meter': pixels_per_meter,
            'pixels_ahead_vehicle': pixels_ahead_vehicle,
        }
        self.birdeye_render = BirdeyeRender(self.world, self.birdeye_params, logger=self.logger)


    def eval(self, configs, data_loader):
        num_finished_scenario = 0
        data_loader.reset_idx_counter()
        while len(data_loader) > 0:
            static_obs = self.env.get_static_obs(configs)
            # sample_scenario_configs被选中的将要测试的场景config文件, num_sampled_scenario为同一个批次采样的场景数量
            selected_scenarios, num_sampled_scenario = data_loader.sampler()
            # 先在这里更新一下configs的data_id
            configs[0].data_id = num_finished_scenario
            num_finished_scenario += num_sampled_scenario

            # 在这里将采样的场景参数传入具体的场景中
            scenario_init_actions = selected_scenarios
            obs, infos = self.env.reset(configs, scenario_init_actions)

            # 从这个函数进入,实际配置ego_vehicle的规控算法
            self.agent_policy.set_ego_and_route(self.env.get_ego_vehicles(), infos, static_obs=static_obs)

            # 定义测试路线得分的list
            score_list = {s_i: [] for s_i in range(num_sampled_scenario)}
            while not self.env.all_scenario_done():
                # get action from agent policy and scenario policy (assume using one batch)
                ego_actions = self.agent_policy.get_action(obs, infos, deterministic=True)
                # 这里研究的对象是测试空间的初始位置分布,而时间序列的对抗性动作是下一篇的工作
                scenario_actions = [None]

                # apply action to env and get obs
                obs, rewards, _, infos = self.env.step(ego_actions=ego_actions, scenario_actions=scenario_actions)

                # save video
                if self.save_video:
                    self.logger.add_frame(pygame.surfarray.array3d(self.display).transpose(1, 0, 2))

                # accumulate scores of corresponding scenario
                reward_idx = 0
                for e_i, _ in enumerate(infos):
                    score = rewards[reward_idx] if self.scenario_category == 'planning' else 1-infos[reward_idx]['iou_loss']
                    score_list[e_i].append(score)
                    reward_idx += 1

            # save video
            if self.save_video:
                data_ids = [config.data_id for config in configs]
                self.logger.save_video(data_ids=data_ids)

            # 打印当前场景的平均reward
            self.logger.log(f'[{num_finished_scenario}/{data_loader.num_total_scenario}] Ranking scores for batch scenario:', 'yellow')
            for s_i in score_list.keys():
                self.logger.log('\t Average reward of Env id ' + str(s_i) + ': ' + str(np.mean(score_list[s_i])), 'yellow')

            # 如果要自定义测试评分,需要从这里进入修改
            data_ids = [config.data_id for config in configs]
            all_running_results = self.logger.add_eval_results_simple(records=self.env.running_results)
            all_scores = get_route_scores_simple(all_running_results, data_ids)
            self.logger.add_eval_results_simple(scores=all_scores)
            self.logger.print_eval_results_simple(data_ids)
            self.logger.save_eval_results()

            # clean up all things
            self.logger.log(">> Current executed scenarios are completed. Clearning up all actors")
            self.env.clean_up()

        # print(f"All scenarios are finished, total {num_finished_scenario} scenarios")


    def run(self, parameters=None):
        config_by_map = scenario_parse_simple(self.scenario_config, self.logger)
        for map in config_by_map.keys():  # map是地图的名称
            # create scenarios within the vectorized wrapper, 含有running_results属性
            self.env = VectorWrapper(
                self.env_params,
                self.scenario_config,
                self.world,
                self.birdeye_render,
                self.display,
                self.logger
            )

            # prepare data loader and buffer, parameters是待测测试参数的组合
            data_loader = ScenarioDataLoaderSimple(parameters, self.num_scenario, map, self.world)

            # run with different modes
            self.agent_policy.load_model()
            self.agent_policy.set_mode('eval')
            self.eval(configs=config_by_map[map], data_loader=data_loader)

        if self.step_by_step:
            # 加载log文件夹下的测试得分并且返回(仅仅用于代理模型的步进)
            found_file_path = os.path.join(self.logger.output_dir, 'eval_results', 'results.pkl')
            init_score, collision = get_test_result(found_file_path, only_latest=True)
            y_init = np.array([init_score]).reshape(-1, )

            # 不仅仅是碰撞,还要筛选那些危险性更大且更稀疏场景
            if collision > 0:
                y = screen_most_risky_scores(init_score, exp_name=self.exp_name, log_base_dir='/home/hp/SENSE/log/baselines')
            else:
                y = y_init

            return y, collision
        else:
            pass

    def close(self):
        pygame.quit() 
        if self.env:
            self.env.clean_up()