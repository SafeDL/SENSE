from gym.envs.registration import register
# 调用register函数来注册一个新的Gym环境,id指定环境的唯一标识符,entry_point指定环境的入口点为safebench.gym_carla.envs模块中的CarlaEnv类
register(
    id='carla-v0',
    entry_point='safebench.gym_carla.envs:CarlaEnv',
)
