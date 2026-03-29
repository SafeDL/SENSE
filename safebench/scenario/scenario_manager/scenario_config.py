"""
1、ScenarioConfig类：包含了一些基本的场景配置参数
2、RouteScenarioConfig类：专门用于配置路线场景,包含其他参与者、触发点、路线变量名等
"""

import carla


class ScenarioConfig(object):
    """
    Configuration of parsed scenario
    """

    auto_ego = False
    num_scenario = None
    route_region = ''
    data_id = 0
    scenario_folder = None
    scenario_id = 0
    route_id = 0
    risk_level = 0
    parameters = None

    town = ''
    name = ''
    weather = None
    scenario_file = None
    initial_transform = None
    initial_pose = None
    trajectory = None
    texture_dir = None


class RouteScenarioConfig(object):
    """
        configuration of a RouteScenario
    """
    other_actors = []
    trigger_points = []
    route_var_name = None
    subtype = None
    parameters = None
    weather = carla.WeatherParameters()
    num_scenario = None
    friction = None


class PerceptionScenarioConfig(object):
    """
        configuration of a PerceptionScenario
    """
    other_actors = []
    trigger_points = []
    route_var_name = None
    subtype = None
    parameters = None
    weather = carla.WeatherParameters()
    num_scenario = None
    friction = None
    ego_id = 0
    texture_dir = None


