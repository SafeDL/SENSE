class BasePolicy:
    name = 'base'
    type = 'unlearnable'

    def __init__(self, config, logger):
        self.ego_vehicles = None

    def set_ego_and_route(self, ego_vehicles, info, static_obs=None):
        self.ego_vehicles = ego_vehicles

    def train(self, replay_buffer):
        raise NotImplementedError()

    def set_mode(self, mode):
        raise NotImplementedError()

    def get_action(self, state, infos, deterministic):
        raise NotImplementedError()

    def load_model(self):
        raise NotImplementedError()

    def save_model(self, episode):
        raise NotImplementedError()
