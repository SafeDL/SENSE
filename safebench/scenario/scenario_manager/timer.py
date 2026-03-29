"""
用于管理和访问carla仿真中的游戏时间
"""

import datetime


class GameTime(object):
    """
        This (static) class provides access to the CARLA game time.
        The elapsed game time can be simply retrieved by calling: GameTime.get_time()
    """

    _current_game_time = 0.0  # 记录从计时器启动后经过的游戏时间
    _carla_time = 0.0  # 记录 CARLA 仿真中的时间
    _last_frame = 0  # 记录最后一帧的编号
    _platform_timestamp = 0  # 记录平台的时间戳
    _init = False  # 计时器是否初始化

    @staticmethod
    def on_carla_tick(timestamp):
        """
            接收 CARLA 的时间戳,只有当帧编号比最后一个帧编号新时才更新时间
        """
        if GameTime._last_frame < timestamp.frame:
            frames = timestamp.frame - GameTime._last_frame if GameTime._init else 1
            GameTime._current_game_time += timestamp.delta_seconds * frames
            GameTime._last_frame = timestamp.frame
            GameTime._platform_timestamp = datetime.datetime.now()
            GameTime._init = True
            GameTime._carla_time = timestamp.elapsed_seconds

    @staticmethod
    def restart():
        """
            重置游戏计时器，将所有时间相关的静态变量重置为0
        """
        GameTime._current_game_time = 0.0
        GameTime._carla_time = 0.0
        GameTime._last_frame = 0
        GameTime._init = False

    @staticmethod
    def get_time():
        """
            Returns elapsed game time
        """
        return GameTime._current_game_time

    @staticmethod
    def get_carla_time():
        """
            Returns elapsed game time
        """
        return GameTime._carla_time

    @staticmethod
    def get_wallclocktime():
        """
            Returns elapsed game time
        """
        return GameTime._platform_timestamp

    @staticmethod
    def get_frame():
        """
            Returns elapsed game time
        """
        return GameTime._last_frame
