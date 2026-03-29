"""
CarlaEnv 类是一个基于 OpenAI Gym 接口的 CARLA 模拟器环境。其设计逻辑和功能可以概括如下:
(1)初始化：
    类的初始化方法 __init__ 接收环境参数 env_params 以及其他可选参数（如 birdeye_render、display、world 和 logger）
    初始化各种传感器（如碰撞传感器、激光雷达传感器和摄像头传感器）和场景管理器 ScenarioManager
    根据不同的场景类别（如 planning、scenic 和 perception）设置观测空间和动作空间
(2)传感器创建：
    方法 _create_sensors 创建并配置碰撞传感器、激光雷达传感器和摄像头传感器
(3)场景创建和运行：
    方法 _create_scenario 根据不同的场景类别创建相应的场景（如 PerceptionScenario、RouteScenario）
    方法 _run_scenario 运行场景管理器中的场景
(4)重置环境：
    方法 reset 重置环境，包括创建传感器、加载和运行场景、附加传感器、解析路线等
(5)步骤执行：
    方法 step_before_tick 在每个tick carla world 之前执行动作，包括计算加速度和转向角度，并应用控制
    方法 step_after_tick 在每个tick carla world 之后更新车辆和行人的多边形信息、车辆状态信息等
(6)观测和奖励：
    方法 _get_obs 获取当前的观测信息，包括状态观测、激光雷达图像、鸟瞰图像和摄像头图像
    方法 _get_reward 计算每个时间步的奖励，奖励函数考虑了碰撞、转向、偏离车道、速度跟踪等因素
(7)其他辅助方法：
    方法 _get_info 获取当前状态的信息
    方法 _init_traffic_light 初始化交通信号灯
    方法 _create_vehicle_blueprint 创建车辆蓝图
    方法 _get_actor_polygons 获取演员的多边形信息
    方法 _get_actor_info 获取演员的状态信息
    方法 _get_cost 计算碰撞的代价
    方法 _terminal 判断场景是否结束
    方法 _remove_sensor 和 _remove_ego 移除传感器和自车
    方法 clean_up 清理环境
"""

import random
import numpy as np
import pygame
from skimage.transform import resize
import gym
from gym import spaces
import carla
from safebench.gym_carla.envs.route_planner import RoutePlanner
from safebench.gym_carla.envs.misc import (
    display_to_rgb, 
    rgb_to_display_surface, 
    get_lane_dis, 
    get_pos, 
    get_preview_lane_dis
)
from safebench.scenario.scenario_definition.route_scenario import RouteScenario
from safebench.scenario.scenario_manager.scenario_manager import ScenarioManager
from safebench.scenario.tools.route_manipulation import interpolate_trajectory
from safebench.scenario.scenario_definition.atomic_criteria import Status
from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider


class CarlaEnv(gym.Env):
    """ 
        An OpenAI-gym style interface for CARLA simulator. 
    """
    def __init__(self, env_params, birdeye_render=None, display=None, world=None, logger=None):
        assert world is not None, "the world passed into CarlaEnv is None"

        self.config = None
        self.world = world
        self.display = display
        self.logger = logger
        self.birdeye_render = birdeye_render

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0
        self.is_running = True
        self.env_id = None
        self.ego_vehicle = None
        self.auto_ego = env_params['auto_ego']

        self.collision_sensor = None
        self.lidar_sensor = None
        self.camera_sensor = None
        self.lidar_data = None
        self.lidar_height = 2.1
        
        # scenario manager
        use_scenic = True if  env_params['scenario_category'] == 'scenic' else False
        self.scenario_manager = ScenarioManager(self.logger, use_scenic=use_scenic)

        # for birdeye view and front view visualization
        self.display_size = env_params['display_size']
        self.obs_range = env_params['obs_range']
        self.d_behind = env_params['d_behind']
        self.disable_lidar = env_params['disable_lidar']

        # for env wrapper
        self.max_past_step = env_params['max_past_step']
        self.max_episode_step = env_params['max_episode_step']
        self.max_waypt = env_params['max_waypt']
        self.lidar_bin = env_params['lidar_bin']
        self.out_lane_thres = env_params['out_lane_thres']
        self.desired_speed = env_params['desired_speed']
        self.acc_max = env_params['continuous_accel_range'][1]
        self.steering_max = env_params['continuous_steer_range'][1]

        # for scenario
        self.ROOT_DIR = env_params['ROOT_DIR']
        self.scenario_category = env_params['scenario_category']  # 指出是进行感知类测试还是规划类测试
        self.warm_up_steps = env_params['warm_up_steps']

        if self.scenario_category in ['planning', 'scenic']:
            self.obs_size = int(self.obs_range/self.lidar_bin)
            """
            字典 observation_space_dict,用于描述观测空间的不同部分。具体来说：
            camera: 定义了一个形状为 (self.obs_size, self.obs_size, 3) 的 Box 空间，表示相机图像的观测空间。图像的每个像素值在 0 到 255 之间，数据类型为 np.uint8
            lidar: 定义了一个形状为 (self.obs_size, self.obs_size, 3) 的 Box 空间，表示激光雷达图像的观测空间。图像的每个像素值在 0 到 255 之间，数据类型为 np.uint8
            birdeye: 定义了一个形状为 (self.obs_size, self.obs_size, 3) 的 Box 空间，表示鸟瞰图的观测空间。图像的每个像素值在 0 到 255 之间，数据类型为 np.uint8
            state: 定义了一个 Box 空间，表示状态信息的观测空间。状态信息的每个值在 [-2, -1, -5, 0] 到 [2, 1, 30, 1] 之间，数据类型为 np.float32
            """
            observation_space_dict = {
                'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
                'lidar': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
                'birdeye': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
                'state': spaces.Box(np.array([-2, -1, -5, 0], dtype=np.float32), np.array([2, 1, 30, 1], dtype=np.float32), dtype=np.float32)
            }
        elif self.scenario_category == 'perception':
            self.obs_size = env_params['image_sz']
            observation_space_dict = {
                'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
            }
        else:
            raise ValueError(f'Unknown scenario category: {self.scenario_category}')

        # define obs space
        self.observation_space = spaces.Dict(observation_space_dict)

        # action and observation spaces
        self.discrete = env_params['discrete']
        self.discrete_act = [env_params['discrete_acc'], env_params['discrete_steer']]  # acc, steer
        self.n_acc = len(self.discrete_act[0])
        self.n_steer = len(self.discrete_act[1])
        if self.discrete:
            self.action_space = spaces.Discrete(self.n_acc * self.n_steer)
        else:
            # assume the output of NN is from -1 to 1
            self.action_space = spaces.Box(np.array([-1, -1], dtype=np.float32), np.array([1, 1], dtype=np.float32), dtype=np.float32)  # acc, steer

    def _create_sensors(self): # 这个函数主要用于配置传感器信息
        # collision sensor
        self.collision_hist_l = 1  # collision history length
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        if self.scenario_category != 'perception':
            # lidar sensor
            self.lidar_trans = carla.Transform(carla.Location(x=0.0, z=self.lidar_height))
            self.lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            self.lidar_bp.set_attribute('channels', '16')
            self.lidar_bp.set_attribute('range', '1000')
        
        # camera sensor
        self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8) 
        self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
        self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
        self.camera_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute('sensor_tick', '0.02')

    def _create_scenario(self, config, env_id):
        self.logger.log(f">> Loading scenario data id: {config.data_id}")

        # create scenario according to different types
        scenario = RouteScenario(
            world=self.world,
            config=config,
            ego_id=env_id,
            max_running_step=self.max_episode_step,
            logger=self.logger
        )

        # init scenario
        self.ego_vehicle = scenario.ego_vehicle
        self.scenario_manager.load_scenario(scenario)

    def _run_scenario(self, scenario_init_action):
        self.scenario_manager.run_scenario(scenario_init_action)

    def _parse_route(self, config):
        # interp waypoints as init waypoints
        route = interpolate_trajectory(self.world, config.trajectory)
        waypoints_list = []
        carla_map = self.world.get_map()
        for node in route:
            loc = node[0].location
            waypoint = carla_map.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            waypoints_list.append(waypoint)
        return waypoints_list

    def get_static_obs(self, config):
        """
            This function returns static observation used for static scenario generation
        """
        # get route
        route = interpolate_trajectory(self.world, config.trajectory, 5.0)

        # get [x, y] along the route
        waypoint_xy = []
        for transform_tuple in route:
            waypoint_xy.append([transform_tuple[0].location.x, transform_tuple[0].location.y])

        # 对于评估而言,使用固定的速度
        state = {
            'route': np.array(waypoint_xy),
            'target_speed': self.desired_speed,
        }

        # 对于场景策的训练而言,速度是随机生成的
        # state = {
        #     'route': np.array(waypoint_xy),
        #     'target_speed': random.uniform(20, 50),
        # }

        return state

    def reset(self, config, env_id, scenario_init_action):
        self.config = config
        self.env_id = env_id

        # create sensors, load and run scenarios
        self._create_sensors()  # 只是指明了传感器的蓝图、生成位置等;
        self._create_scenario(config, env_id)  # 创建主车、要执行的场景类的实例、评分准则
        # 实际上会调用scenario_manager执行run,而这个run函数中调用具体的场景实例,创建场景中对抗actor
        self._run_scenario(scenario_init_action)
        self._attach_sensor()  # 在这里才真正生成了传感器

        # route planner for ego vehicle
        self.route_waypoints = self._parse_route(config)  # 由scenario_config定义的测试轨迹插值路径点
        self.routeplanner = RoutePlanner(self.ego_vehicle, self.max_waypt, self.route_waypoints)
        self.waypoints, _, _, _, _, self.vehicle_front, = self.routeplanner.run_step()  # waypoints为由routeplanner计算的buffer_size路径点

        # Get actors polygon list (for visualization)
        self.vehicle_polygons = [self._get_actor_polygons('vehicle.*')]
        self.walker_polygons = [self._get_actor_polygons('walker.*')]

        # 获取carla server端的 actors info list
        vehicle_info_dict_list = self._get_actor_info('vehicle.*')
        self.vehicle_trajectories = [vehicle_info_dict_list[0]]
        self.vehicle_accelerations = [vehicle_info_dict_list[1]]
        self.vehicle_angular_velocities = [vehicle_info_dict_list[2]]
        self.vehicle_velocities = [vehicle_info_dict_list[3]]

        # Update timesteps
        self.time_step = 0
        self.reset_step += 1

        # applying setting can tick the world and get data from sensros
        # removing this block will cause error: AttributeError: 'NoneType' object has no attribute 'raw_data'
        self.settings = self.world.get_settings()
        self.world.apply_settings(self.settings)

        for _ in range(self.warm_up_steps):
            self.world.tick()
        return self._get_obs(), self._get_info()

    def _attach_sensor(self):
        # Add collision sensor
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego_vehicle)
        self.collision_sensor.listen(lambda event: get_collision_hist(event))

        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)
        self.collision_hist = []

        # Add lidar sensor
        if self.scenario_category != 'perception' and not self.disable_lidar:
            self.lidar_sensor = self.world.spawn_actor(self.lidar_bp, self.lidar_trans, attach_to=self.ego_vehicle)
            self.lidar_sensor.listen(lambda data: get_lidar_data(data))

        def get_lidar_data(data):
            self.lidar_data = data

        # Add camera sensor
        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego_vehicle)
        self.camera_sensor.listen(lambda data: get_camera_img(data))

        def get_camera_img(data):            
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_img = array

    def step_before_tick(self, ego_action, scenario_action):
        if self.world:
            snapshot = self.world.get_snapshot()
            if snapshot:
                timestamp = snapshot.timestamp
                # get update on evaluation results before getting update of running status
                if self.scenario_category in ['perception']:
                    assert isinstance(ego_action, dict), 'ego action in ObjectDetectionScenario should be a dict'
                    world_2_camera = np.array(self.camera_sensor.get_transform().get_inverse_matrix())
                    fov = self.camera_bp.get_attribute('fov').as_float()
                    image_w, image_h = self.obs_size, self.obs_size
                    self.scenario_manager.background_scenario.evaluate(ego_action, world_2_camera, image_w, image_h, fov, self.camera_img)
                    ego_action = ego_action['ego_action']

                # 每一次在tick carla world之前,都会通过scenario_manager检查场景状态
                # 如果达到了触发条件,那么具体场景会被触发并执行一个scenario action
                self.scenario_manager.get_update(timestamp, scenario_action)
                self.is_running = self.scenario_manager._running

                # Calculate acceleration and steering
                if not self.auto_ego:
                    if self.discrete:
                        acc = self.discrete_act[0][ego_action // self.n_steer]
                        steer = self.discrete_act[1][ego_action % self.n_steer]
                    else:
                        acc = ego_action[0]
                        steer = ego_action[1]

                    # normalize and clip the action
                    acc = acc * self.acc_max
                    steer = steer * self.steering_max
                    acc = max(min(self.acc_max, acc), -self.acc_max)
                    steer = max(min(self.steering_max, steer), -self.steering_max)

                    # Convert acceleration to throttle and brake
                    if acc > 0:
                        throttle = np.clip(acc / 3, 0, 1)
                        brake = 0
                    else:
                        throttle = 0
                        brake = np.clip(-acc / 8, 0,  1)

                    # apply control
                    act = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
                    self.ego_vehicle.apply_control(act)
            else:
                self.logger.log('>> Can not get snapshot!', color='red')
                raise Exception()
        else:
            self.logger.log('>> Please specify a Carla world!', color='red')
            raise Exception()

    def step_after_tick(self):
        # Append actors polygon list
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        while len(self.vehicle_polygons) > self.max_past_step:
            self.vehicle_polygons.pop(0)
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)
        while len(self.walker_polygons) > self.max_past_step:
            self.walker_polygons.pop(0)

        # Append actors info list
        vehicle_info_dict_list = self._get_actor_info('vehicle.*')
        self.vehicle_trajectories.append(vehicle_info_dict_list[0])
        while len(self.vehicle_trajectories) > self.max_past_step:
            self.vehicle_trajectories.pop(0)
        self.vehicle_accelerations.append(vehicle_info_dict_list[1])
        while len(self.vehicle_accelerations) > self.max_past_step:
            self.vehicle_accelerations.pop(0)
        self.vehicle_angular_velocities.append(vehicle_info_dict_list[2])
        while len(self.vehicle_angular_velocities) > self.max_past_step:
            self.vehicle_angular_velocities.pop(0)
        self.vehicle_velocities.append(vehicle_info_dict_list[3])
        while len(self.vehicle_velocities) > self.max_past_step:
            self.vehicle_velocities.pop(0)

        # route planner
        self.waypoints, _, _, _, _, self.vehicle_front, = self.routeplanner.run_step()

        # Update timesteps
        self.time_step += 1
        self.total_step += 1
        # 用于train_scenario_agent
        return (self._get_obs(), self._get_scenario_reward(), self._terminal(), self._get_info())


    def _get_info(self):
        # state information
        info = {
            'waypoints': self.waypoints,
            'route_waypoints': self.route_waypoints,
            'vehicle_front': self.vehicle_front,
            'cost': self._get_cost()
        }

        # info from scenarios
        info.update(self.scenario_manager.background_scenario.update_info())
        return info

    def _init_traffic_light(self):
        actor_list = self.world.get_actors()
        for actor in actor_list:
            if isinstance(actor, carla.TrafficLight):
                actor.set_red_time(3)
                actor.set_green_time(3)
                actor.set_yellow_time(1)

    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _get_actor_polygons(self, filt):
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            # Get x, y and yaw of the actor
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            # Get length and width
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    def _get_actor_info(self, filt):
        actor_trajectory_dict = {}
        actor_acceleration_dict = {}
        actor_angular_velocity_dict = {}
        actor_velocity_dict = {}

        for actor in self.world.get_actors().filter(filt):
            actor_trajectory_dict[actor.id] = actor.get_transform()
            actor_acceleration_dict[actor.id] = actor.get_acceleration()
            actor_angular_velocity_dict[actor.id] = actor.get_angular_velocity()
            actor_velocity_dict[actor.id] = actor.get_velocity()
        return actor_trajectory_dict, actor_acceleration_dict, actor_angular_velocity_dict, actor_velocity_dict

    def _get_obs(self):
        """获取当前状态的强化学习式的观察描述,并且同时更新BEV的视图渲染"""
        ego_trans = self.ego_vehicle.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_yaw = ego_trans.rotation.yaw / 180 * np.pi
        # 车辆当前位置与规划轨迹中线之间的横向距离。这个值用于评估车辆是否偏离车道中心
        lateral_dis, w = get_preview_lane_dis(self.waypoints, ego_x, ego_y)
        yaw = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])
        # 表示车辆当前朝向与车道方向之间的偏差角度。这个值用于评估车辆是否偏离车道方向
        delta_yaw = np.arcsin(np.cross(w, yaw))

        v = self.ego_vehicle.get_velocity()
        speed = np.sqrt(v.x**2 + v.y**2) # 表示车辆的当前速度(m/s)这个值用于评估车辆的行驶速度
        acc = self.ego_vehicle.get_acceleration()
        state = np.array([lateral_dis, -delta_yaw, speed, self.vehicle_front])  # self.vehicle_front表示车辆前方的距离或状态信息。这个值用于评估车辆前方的情况，例如前方是否有障碍物或其他车辆

        if self.scenario_category != 'perception': 
            # set ego information for birdeye_render
            self.birdeye_render.set_hero(self.ego_vehicle, self.ego_vehicle.id)
            self.birdeye_render.vehicle_polygons = self.vehicle_polygons
            self.birdeye_render.walker_polygons = self.walker_polygons
            self.birdeye_render.waypoints = self.waypoints  # 供给BEV render的路径是由route_planner规划出来的

            # render birdeye image with the birdeye_render
            birdeye_render_types = ['roadmap', 'actors', 'waypoints']
            birdeye_surface = self.birdeye_render.render(birdeye_render_types)
            birdeye_surface = pygame.surfarray.array3d(birdeye_surface)
            center = (int(birdeye_surface.shape[0]/2), int(birdeye_surface.shape[1]/2))
            width = height = int(self.display_size/2)
            birdeye = birdeye_surface[center[0]-width:center[0]+width, center[1]-height:center[1]+height]
            birdeye = display_to_rgb(birdeye, self.obs_size)

            if not self.disable_lidar:
                # get Lidar image
                point_cloud = np.copy(np.frombuffer(self.lidar_data.raw_data, dtype=np.dtype('f4')))
                point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 4), 4))
                x = point_cloud[:, 0:1]
                y = point_cloud[:, 1:2]
                z = point_cloud[:, 2:3]
                intensity = point_cloud[:, 3:4]
                point_cloud = np.concatenate([y, -x, z], axis=1)
                # Separate the 3D space to bins for point cloud, x and y is set according to self.lidar_bin, and z is set to be two bins.
                y_bins = np.arange(-(self.obs_range - self.d_behind), self.d_behind + self.lidar_bin, self.lidar_bin)
                x_bins = np.arange(-self.obs_range / 2, self.obs_range / 2 + self.lidar_bin, self.lidar_bin)
                z_bins = [-self.lidar_height - 1, -self.lidar_height + 0.25, 1]
                # Get lidar image according to the bins
                lidar, _ = np.histogramdd(point_cloud, bins=(x_bins, y_bins, z_bins))
                lidar[:, :, 0] = np.array(lidar[:, :, 0] > 0, dtype=np.uint8)
                lidar[:, :, 1] = np.array(lidar[:, :, 1] > 0, dtype=np.uint8)
                wayptimg = birdeye[:, :, 0] < 0  # Equal to a zero matrix
                wayptimg = np.expand_dims(wayptimg, axis=2)
                wayptimg = np.fliplr(np.rot90(wayptimg, 3))
                # Get the final lidar image
                lidar = np.concatenate((lidar, wayptimg), axis=2)
                lidar = np.flip(lidar, axis=1)
                lidar = np.rot90(lidar, 1) * 255

                # display birdeye image
                birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
                self.display.blit(birdeye_surface, (0, self.env_id*self.display_size))

                # display lidar image
                lidar_surface = rgb_to_display_surface(lidar, self.display_size)
                self.display.blit(lidar_surface, (self.display_size, self.env_id*self.display_size))

                # display camera image
                camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
                camera_surface = rgb_to_display_surface(camera, self.display_size)
                self.display.blit(camera_surface, (self.display_size*2, self.env_id*self.display_size))
            else:
                # display birdeye image
                birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
                # birdeye_surface是一个表示鸟瞰图的pygame.Surface的对象
                # (0, self.env_id*self.display_size)元素表示将 birdeye_surface 绘制到 self.display 上的坐标位置
                # 0 表示 x 坐标，self.env_id*self.display_size 表示 y 坐标。self.env_id 是环境的 ID，self.display_size 是显示的大小
                self.display.blit(birdeye_surface, (0, self.env_id*self.display_size))

                # display camera image
                camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
                camera_surface = rgb_to_display_surface(camera, self.display_size)
                self.display.blit(camera_surface, (self.display_size, self.env_id*self.display_size))

            obs = {
                'camera': camera.astype(np.uint8),
                'lidar': None if self.disable_lidar else lidar.astype(np.uint8),
                'birdeye': birdeye.astype(np.uint8),
                'state': state.astype(np.float32),
            }
        else:
            """ Get the observations for object detection. """
            camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
            camera_surface = rgb_to_display_surface(camera, self.display_size)
            self.display.blit(camera_surface, (0, self.env_id*self.display_size))

            obs = {
                'camera': camera.astype(np.uint8),
                'state': state.astype(np.float32),
            }
        return obs

    def _get_scenario_reward(self):
        """
        对于场景策略而言, REINFORCE奖励设计如下：
        1. 发生碰撞奖励为正（仿真一旦碰撞即终止, r_collision只会在碰撞帧为正）
        2. 距离越近奖励越大
        3. 合理性/约束惩罚
        """
        # 1. 碰撞奖励 (Collision Bonus, r_b)
        is_collision = self.scenario_manager.running_record[-1]['collision'] == Status.FAILURE
        r_collision = 25.0 if is_collision else 0.0

        # 2. 距离奖励 (Distance Metric, -r_d) [cite: 216-217]
        # 论文直接使用欧氏距离。为了数值稳定性，我们可以将其转化为归一化的接近度。
        ego_loc = self.ego_vehicle.get_location()
        adv_veh = self.scenario_manager.scenario_list[0].reference_actor

        if adv_veh is not None:
            dist = ego_loc.distance(adv_veh.get_location())
        else:
            dist = 100.0  # 默认大距离

        # 将距离转化为 [0, 1] 的奖励，距离越近奖励越大
        # 使用 exponential kernel 比线性更好，避免远距离时的梯度消失
        k_dist = 0.2
        r_distance = np.exp(-k_dist * dist) * 5.0  # 系数可调，作为 dense reward

        # 3. 合理性/约束惩罚 (Penalty, r_p)
        # 论文检查障碍物是否生成得离 route 太近（指初始生成的瞬间）。
        # 如果是在运行过程中的 step reward，我们可以惩罚“不动的障碍物”或者“瞬移”。
        # 这里我们简化为：如果碰撞发生了，但相对速度极低（像是故意挡路而不是撞击），给予惩罚。
        r_penalty = 0.0

        # 组合奖励
        # 注意：论文中 r_d 是负项，意味着我们要最小化距离。
        # 这里我们用 r_distance (越大越好) 替代了 -r_d。
        total_reward = r_collision + r_distance - r_penalty

        return total_reward


    def _get_ego_reward(self):
        """
        训练ego_agent的奖励函数
        It penalizes the agent for collision, out of lane, too fast, and too much steering.
        The final reward is a weighted sum of these components, encouraging smooth, safe, and lane-compliant driving.
        """
        r_collision = -1 if len(self.collision_hist) > 0 else 0

        # reward for steering:
        r_steer = -self.ego_vehicle.get_control().steer ** 2

        # reward for out of lane
        ego_x, ego_y = get_pos(self.ego_vehicle)
        dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)  # w 是车道方向向量，用于将当前速度投影到车道方向
        r_out = -1 if abs(dis) > self.out_lane_thres else 0

        # reward for speed tracking
        v = self.ego_vehicle.get_velocity()

        # cost for too fast
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)  # m/s
        r_fast = -1 if lspeed_lon > self.desired_speed / 3.6 else 0

        # cost for lateral acceleration
        r_lat = -abs(self.ego_vehicle.get_control().steer) * lspeed_lon**2

        # combine all rewards
        r = 1 * r_collision + 1 * lspeed_lon + 10 * r_fast + 1 * r_out + r_steer * 5 + 0.2 * r_lat
        return r

    def _get_cost(self):
        """返回碰撞代价,如果发生了碰撞,将碰撞代价设为-1,否则为0"""
        r_collision = 0
        if len(self.collision_hist) > 0:
            r_collision = -1
        return r_collision

    def _terminal(self):
        return not self.scenario_manager._running 

    def _remove_sensor(self):
        if self.collision_sensor is not None:
            self.collision_sensor.stop()
            self.collision_sensor.destroy()
            self.collision_sensor = None
        if self.lidar_sensor is not None:
            self.lidar_sensor.stop()
            self.lidar_sensor.destroy()
            self.lidar_sensor = None
        if self.camera_sensor is not None:
            self.camera_sensor.stop()
            self.camera_sensor.destroy()
            self.camera_sensor = None

    def _remove_ego(self):
        # TODO: ego can be reused.
        if self.ego_vehicle is not None:
            self.ego_vehicle.destroy()
            self.ego_vehicle = None

    def clean_up(self):
        self._remove_sensor()
        if self.scenario_category != 'scenic':
            self._remove_ego()
        self.scenario_manager.clean_up()
