"""
管理动态场景，包括初始化、触发、更新和停止场景
"""

from safebench.scenario.scenario_manager.carla_data_provider import CarlaDataProvider
from safebench.scenario.scenario_manager.timer import GameTime
from safebench.scenario.tools.scenario_utils import calculate_distance_locations


class ScenarioManager(object):
    """
        Dynamic version scenario manager class. This class holds all functionality
        required to initialize, trigger, update and stop a scenario.
    """

    def __init__(self, logger, use_scenic=False):
        self.logger = logger
        self.scenic = use_scenic
        self._reset()

    def _reset(self):
        # 重置场景管理器的状态,包括背景场景(亦即RouteScenario的实例)、ego车辆、场景列表、触发的场景集合、运行状态、时间戳和运行记录
        self.background_scenario = None
        self.ego_vehicle = None
        self.scenario_list = None
        self.triggered_scenario = set()  # 存储已经被触发的场景
        self._running = False
        self._timestamp_last_run = 0.0
        self.cur_distance = None  # 记录ego车辆和reference_actor之间的距离
        self.running_record = []
        GameTime.restart()

    def clean_up(self):
        if self.background_scenario is not None:
            self.background_scenario.clean_up()

    def load_scenario(self, scenario):
        # 加载给定的场景，重置状态，并设置背景场景、ego车辆和场景列表
        self._reset()
        self.background_scenario = scenario
        self.ego_vehicle = scenario.ego_vehicle
        self.scenario_list = scenario.list_scenarios

    def run_scenario(self, scenario_init_action):
        # 这是第一次将场景初始动作传入,将运行状态标记为True
        self._running = True
        self._init_scenarios(scenario_init_action)

    def _init_scenarios(self, scenario_init_action):
        # 生成场景中背景车
        self.background_scenario.initialize_actors()
        
        # running_scenario代表需要执行的具体场景定义,比如DynamicObjectCrossing, OtherLeadingVehicle等
        for running_scenario in self.scenario_list:
            # some scenario passes actions when creating behavior
            running_scenario.create_behavior(scenario_init_action)
            # init actors after passing in init actions
            running_scenario.initialize_actors()
    
    def stop_scenario(self):
        self._running = False

    def update_running_status(self):
        # 更新场景的运行状态，记录运行记录，并根据需要停止场景
        record, stop = self.background_scenario.get_running_status(self.running_record, self.cur_distance)
        self.running_record.append(record)
        if stop:
            self._running = False

    def get_update(self, timestamp, scenario_action):
        # 根据时间戳和场景动作更新场景状态,包括触发场景和更新触发场景的行为
        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds
            GameTime.on_carla_tick(timestamp)
            CarlaDataProvider.on_carla_tick()  # 在每一次tick的时候,会进入更新地图中的actor的速度、位置等信息

            # check whether the scenario should be triggered
            for spawned_scenario in self.scenario_list:
                ego_location = CarlaDataProvider.get_location(self.ego_vehicle)
                self.cur_distance = None
                if spawned_scenario.reference_actor:  # 以DynamicObjectCrossing为例,这里的reference_actor是walker
                    reference_location = CarlaDataProvider.get_location(spawned_scenario.reference_actor)
                    self.cur_distance = calculate_distance_locations(ego_location, reference_location)

                if self.cur_distance and self.cur_distance < spawned_scenario.trigger_distance_threshold:
                    if spawned_scenario not in self.triggered_scenario:
                        self.logger.log(">> Trigger scenario: " + spawned_scenario.name)
                        self.triggered_scenario.add(spawned_scenario)

                # update behavior of triggered scenarios
                for running_scenario in self.triggered_scenario: 
                    # update behavior of carla_agents in scenario
                    running_scenario.update_behavior(scenario_action)

            self.update_running_status()