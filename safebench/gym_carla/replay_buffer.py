import numpy as np
import torch


class RouteReplayBuffer:
    """
        This buffer supports parallel storing transitions from multiple trajectories.
    """
    
    def __init__(self, num_scenario, mode, buffer_capacity=1000):
        self.mode = mode
        self.buffer_capacity = 0.5 * buffer_capacity
        self.num_scenario = num_scenario
        self.buffer_len = 0

        # 高奖励回报相关
        self.reward_threshold = 0.65  # 高奖励回报的阈值
        self.high_reward_capacity = buffer_capacity  # 高奖励回报的最大容量

        # buffers for step info
        self.reset_buffer()

        # buffers for init info
        self.reset_init_buffer()

    def reset_buffer(self):
        self.buffer_ego_actions = [[] for _ in range(self.num_scenario)]
        self.buffer_scenario_actions = [[] for _ in range(self.num_scenario)]
        self.buffer_obs = [[] for _ in range(self.num_scenario)]
        self.buffer_next_obs = [[] for _ in range(self.num_scenario)]
        self.buffer_rewards = [[] for _ in range(self.num_scenario)]
        self.buffer_dones = [[] for _ in range(self.num_scenario)]
        self.buffer_additional_dict = [{} for _ in range(self.num_scenario)]

        # 高奖励缓冲
        self.high_ego_actions = [[] for _ in range(self.num_scenario)]
        self.high_scenario_actions = [[] for _ in range(self.num_scenario)]
        self.high_obs = [[] for _ in range(self.num_scenario)]
        self.high_next_obs = [[] for _ in range(self.num_scenario)]
        self.high_rewards = [[] for _ in range(self.num_scenario)]
        self.high_buffer_dones = [[] for _ in range(self.num_scenario)]
        self.high_additional_dict = [{} for _ in range(self.num_scenario)]


    def reset_init_buffer(self):
        self.buffer_static_obs = []
        self.buffer_init_action = []
        self.buffer_episode_reward = []
        self.buffer_init_additional_dict = {}

        # 高风险初始化缓冲
        self.high_buffer_static_obs = []
        self.high_buffer_init_action = []
        self.high_buffer_episode_reward = []
        self.high_buffer_init_additional_dict = {}


    def store(self, data_list, additional_dict):
        ego_actions = data_list[0]  # data_list的存储分别是[ego_actions, scenario_actions, obs, next_obs, rewards, dones]
        scenario_actions = data_list[1]
        obs = data_list[2]
        next_obs = data_list[3]
        rewards = data_list[4]
        dones = data_list[5]
        self.buffer_len += len(rewards)

        # separate trajectories according to infos
        for s_i in range(len(additional_dict)):
            sid = additional_dict[s_i]['s_id']

            # 判断是否高奖励样本
            if rewards[s_i] >= self.reward_threshold:
                self.high_ego_actions[sid].append(ego_actions[s_i])
                self.high_scenario_actions[sid].append(scenario_actions[s_i])
                self.high_obs[sid].append(obs[s_i])
                self.high_next_obs[sid].append(next_obs[s_i])
                self.high_rewards[sid].append(rewards[s_i])
                self.high_buffer_dones[sid].append(dones[s_i])  # 添加完成标志

                for key in additional_dict[s_i].keys():
                    if key == 's_id':
                        continue
                    if key not in self.high_additional_dict[sid].keys():
                        self.high_additional_dict[sid][key] = []
                    self.high_additional_dict[sid][key].append(additional_dict[s_i][key])

                # 高奖励容量限制
                while len(self.high_rewards[sid]) > self.high_reward_capacity:
                    self.high_ego_actions[sid].pop(0)
                    self.high_scenario_actions[sid].pop(0)
                    self.high_obs[sid].pop(0)
                    self.high_next_obs[sid].pop(0)
                    self.high_rewards[sid].pop(0)
                    self.high_buffer_dones[sid].pop(0)
                    for key in additional_dict[s_i].keys():
                        if key == 's_id':
                            continue
                        self.high_additional_dict[sid][key].pop(0)
            else:
                self.buffer_ego_actions[sid].append(ego_actions[s_i])
                self.buffer_scenario_actions[sid].append(scenario_actions[s_i])
                self.buffer_obs[sid].append(obs[s_i])
                self.buffer_next_obs[sid].append(next_obs[s_i])
                self.buffer_rewards[sid].append(rewards[s_i])
                self.buffer_dones[sid].append(dones[s_i])  # 添加完成标志
                for key in additional_dict[s_i].keys():
                    if key == 's_id':
                        continue
                    if key not in self.buffer_additional_dict[sid].keys():
                        self.buffer_additional_dict[sid][key] = []
                    self.buffer_additional_dict[sid][key].append(additional_dict[s_i][key])
                # 普通样本容量限制
                while len(self.buffer_rewards[sid]) > 0.7 * self.buffer_capacity:
                    self.buffer_ego_actions[sid].pop(0)
                    self.buffer_scenario_actions[sid].pop(0)
                    self.buffer_obs[sid].pop(0)
                    self.buffer_next_obs[sid].pop(0)
                    self.buffer_rewards[sid].pop(0)
                    self.buffer_dones[sid].pop(0)
                    for key in additional_dict[s_i].keys():
                        if key == 's_id':
                            continue
                        self.buffer_additional_dict[sid][key].pop(0)

    def store_init(self, data_list, additional_dict=None):
        static_obs = data_list[0]  # 存储的是测试路线waypoints
        scenario_init_action = data_list[1]  # 存储的是scenario agent给出的初始场景参数
        episode_rewards = data_list[2]  # 存储的是本次轨迹的所有奖励
        for s_i in range(self.num_scenario):
            # 判断是否为高风险轨迹
            if np.mean(episode_rewards) >= self.reward_threshold:
                self.high_buffer_init_action.append(scenario_init_action[s_i])
                self.high_buffer_static_obs.append(static_obs[s_i])
                self.high_buffer_episode_reward.append(np.mean(episode_rewards))

                if additional_dict:
                    for key in additional_dict.keys():
                        if key not in self.high_buffer_init_additional_dict.keys():
                            self.high_buffer_init_additional_dict[key] = []
                        self.high_buffer_init_additional_dict[key].append(additional_dict[key][s_i])
                # 高风险容量限制
                while len(self.high_buffer_init_action) > self.high_reward_capacity:
                    self.high_buffer_init_action.pop(0)
                    self.high_buffer_static_obs.pop(0)
                    self.high_buffer_episode_reward.pop(0)
                    for key in additional_dict.keys():
                        if key not in self.high_buffer_init_additional_dict.keys():
                            continue
                        self.high_buffer_init_additional_dict[key].pop(0)
            else:
                self.buffer_init_action.append(scenario_init_action[s_i])
                self.buffer_static_obs.append(static_obs[s_i])
                self.buffer_episode_reward.append(np.mean(episode_rewards))

                # store additional information in given dict
                if additional_dict:
                    for key in additional_dict.keys():
                        if key not in self.buffer_init_additional_dict.keys():
                            self.buffer_init_additional_dict[key] = []
                        for v in additional_dict[key]:
                            self.buffer_init_additional_dict[key].append(v)
                # 容量限制
                while len(self.buffer_init_action) > self.buffer_capacity:
                    self.buffer_init_action.pop(0)
                    self.buffer_static_obs.pop(0)
                    self.buffer_episode_reward.pop(0)
                    for key in additional_dict.keys():
                        if key not in self.buffer_init_additional_dict.keys():
                            continue
                        self.buffer_init_additional_dict[key].pop(0)

    def sample_init(self, batch_size):
        low_start_idx = 0
        high_start_idx = 0

        low_indices = list(range(low_start_idx, len(self.buffer_init_action)))
        high_indices = list(range(high_start_idx, len(self.high_buffer_init_action)))

        # 高风险采样数量
        high_count = 16

        if len(self.high_buffer_init_action) > high_count:
            high_sample_idx = np.random.choice(high_indices, high_count, replace=False)
        else:
            high_sample_idx = np.random.choice(high_indices, len(self.high_buffer_init_action), replace=False)
        low_count = batch_size - len(high_sample_idx)
        low_sample_idx = np.random.choice(low_indices, low_count, replace=False)

        # 拼接数据
        prepared_static_obs = [self.buffer_static_obs[i] for i in low_sample_idx] + \
                               [self.high_buffer_static_obs[i] for i in high_sample_idx]
        prepared_init_action = [self.buffer_init_action[i] for i in low_sample_idx] + \
                               [self.high_buffer_init_action[i] for i in high_sample_idx]
        prepared_episode_reward = [self.buffer_episode_reward[i] for i in low_sample_idx] + \
                                  [self.high_buffer_episode_reward[i] for i in high_sample_idx]
        prepared_additional_dict = {
            key: [self.buffer_init_additional_dict[key][i] for i in low_sample_idx] + \
                 [self.high_buffer_init_additional_dict[key][i] for i in high_sample_idx]
            for key in set(list(self.buffer_init_additional_dict.keys()) +
                           list(self.high_buffer_init_additional_dict.keys()))}

        static_obs = np.array(prepared_static_obs)
        init_action = np.array(prepared_init_action)
        episode_reward = np.array(prepared_episode_reward)
        batch = {
            'static_obs': static_obs,
            'init_action': init_action,
            'episode_reward': episode_reward,
        }

        # add additional information to the batch (assume with torch)
        for key in self.buffer_init_additional_dict.keys():
            batch[key] = torch.stack(prepared_additional_dict[key])
        return batch

    def sample(self, batch_size):
        # prepare concatenated list
        prepared_ego_actions = []
        prepared_scenario_actions = []
        prepared_obs = []
        prepared_next_obs = []
        prepared_rewards = []
        prepared_dones = []
        prepared_infos = {}

        for s_i in range(self.num_scenario):
            # 选择最近的普通样本
            start_idx = 0
            prepared_ego_actions += self.buffer_ego_actions[s_i][start_idx:]
            prepared_scenario_actions += self.buffer_scenario_actions[s_i][start_idx:]
            prepared_obs += self.buffer_obs[s_i][start_idx:]
            prepared_next_obs += self.buffer_next_obs[s_i][start_idx:]
            prepared_rewards += self.buffer_rewards[s_i][start_idx:]
            prepared_dones += self.buffer_dones[s_i][start_idx:]
            for k_i in self.buffer_additional_dict[s_i].keys():
                if k_i not in prepared_infos.keys():
                    prepared_infos[k_i] = []
                prepared_infos[k_i] += self.buffer_additional_dict[s_i][k_i][start_idx:]

            # 选择高奖励样本
            prepared_ego_actions += self.high_ego_actions[s_i]
            prepared_scenario_actions += self.high_scenario_actions[s_i]
            prepared_obs += self.high_obs[s_i]
            prepared_next_obs += self.high_next_obs[s_i]
            prepared_rewards += self.high_rewards[s_i]
            prepared_dones += self.high_buffer_dones[s_i]
            for k_i in self.high_additional_dict[s_i].keys():
                prepared_infos.setdefault(k_i, []).extend((self.high_additional_dict[s_i][k_i]))

        # prepare batch
        sample_index = np.random.randint(1, len(prepared_rewards), size=batch_size)
        action = prepared_ego_actions if self.mode == 'train_agent' else prepared_scenario_actions
        batch = {
            'action': np.stack(action)[sample_index], # action
            'state': np.stack(prepared_obs)[sample_index, :],         # state
            'n_state': np.stack(prepared_next_obs)[sample_index, :],  # next state
            'reward': np.stack(prepared_rewards)[sample_index],       # reward
            'done': np.stack(prepared_dones)[sample_index],           # done
        }

        # add additional information to the batch
        batch_info = {} 
        for k_i in prepared_infos.keys():
            if k_i == 'route_waypoints':
                continue
            batch_info[k_i] = np.stack(prepared_infos[k_i])[sample_index-1]
            batch_info['n_' + k_i] = np.stack(prepared_infos[k_i])[sample_index]

        # combine two dicts
        batch.update(batch_info)
        return batch


class RouteReplayBuffer_reinforce:
    """
    On-Policy Buffer for REINFORCE.
    Stores transitions for the current epoch and clears them after update.
    """

    def __init__(self, num_scenario, mode, buffer_capacity=1000):
        self.mode = mode
        self.buffer_capacity = buffer_capacity
        self.num_scenario = num_scenario

        # 记录当前buffer中存储的样本数量
        self.buffer_len = 0

        # 初始化 step-wise buffer (如果需要训练ego agent)
        self.reset_buffer()

        # 初始化 episode-wise buffer (核心：用于Scenario REINFORCE训练)
        self.reset_init_buffer()

    def reset_buffer(self):
        """清空 Step 级缓冲区"""
        self.buffer_ego_actions = [[] for _ in range(self.num_scenario)]
        self.buffer_scenario_actions = [[] for _ in range(self.num_scenario)]
        self.buffer_obs = [[] for _ in range(self.num_scenario)]
        self.buffer_next_obs = [[] for _ in range(self.num_scenario)]
        self.buffer_rewards = [[] for _ in range(self.num_scenario)]
        self.buffer_dones = [[] for _ in range(self.num_scenario)]
        self.buffer_additional_dict = [{} for _ in range(self.num_scenario)]

    def reset_init_buffer(self):
        """
        清空 Episode 级缓冲区。
        在 REINFORCE 中，每次 update 结束后必须调用此函数。
        """
        self.buffer_static_obs = []  # 场景输入的静态特征 (Route points)
        self.buffer_init_action = []  # 模型生成的参数 (Action)
        self.buffer_episode_reward = []  # 整条轨迹的奖励 (Return)
        self.buffer_init_additional_dict = {}  # 存储 log_prob, entropy 等关键数据

    def store(self, data_list, additional_dict):
        """
        存储 Step 级数据 (用于 Ego Agent 训练)
        """
        ego_actions, scenario_actions, obs, next_obs, rewards, dones = data_list
        self.buffer_len += len(rewards)

        for s_i in range(len(additional_dict)):
            sid = additional_dict[s_i]['s_id']

            # 直接存储，不做筛选
            self.buffer_ego_actions[sid].append(ego_actions[s_i])
            self.buffer_scenario_actions[sid].append(scenario_actions[s_i])
            self.buffer_obs[sid].append(obs[s_i])
            self.buffer_next_obs[sid].append(next_obs[s_i])
            self.buffer_rewards[sid].append(rewards[s_i])
            self.buffer_dones[sid].append(dones[s_i])

            # 存储额外信息
            for key in additional_dict[s_i].keys():
                if key == 's_id': continue
                if key not in self.buffer_additional_dict[sid]:
                    self.buffer_additional_dict[sid][key] = []
                self.buffer_additional_dict[sid][key].append(additional_dict[s_i][key])

            # 简单的 FIFO 容量控制 (防止内存溢出，虽然 On-policy 通常不会存满)
            if len(self.buffer_rewards[sid]) > self.buffer_capacity:
                self.buffer_ego_actions[sid].pop(0)
                self.buffer_scenario_actions[sid].pop(0)
                self.buffer_obs[sid].pop(0)
                self.buffer_next_obs[sid].pop(0)
                self.buffer_rewards[sid].pop(0)
                self.buffer_dones[sid].pop(0)
                for key in self.buffer_additional_dict[sid]:
                    self.buffer_additional_dict[sid][key].pop(0)

    def store_init(self, data_list, additional_dict=None):
        """
        存储 Episode 级数据 (核心：用于 Scenario Agent 的 REINFORCE 更新)
        """
        static_obs = data_list[0]  # [Num_Scenario, Feature_Dim]
        scenario_init_action = data_list[1]  # [Num_Scenario, Action_Dim]
        episode_rewards = data_list[2]  # [Num_Scenario] (通常是一次 Episode 的平均或总和)

        for s_i in range(self.num_scenario):
            # 1. 存储状态、动作、奖励
            self.buffer_static_obs.append(static_obs[s_i])
            self.buffer_init_action.append(scenario_init_action[s_i])

            # 注意：这里假设传入的 episode_rewards 已经是标量或者需要取均值
            # 如果 REINFORCE 需要整条轨迹的回报，这里直接存数值即可
            r = np.mean(episode_rewards) if isinstance(episode_rewards, (list, np.ndarray)) else episode_rewards
            self.buffer_episode_reward.append(r)

            # 2. 存储 log_prob, entropy 等 (这对于计算梯度至关重要)
            if additional_dict:
                for key in additional_dict.keys():
                    if key not in self.buffer_init_additional_dict:
                        self.buffer_init_additional_dict[key] = []
                    # 假设 additional_dict[key] 是一个 tensor 或 list，取第 s_i 个元素
                    self.buffer_init_additional_dict[key].append(additional_dict[key][s_i])

    @property
    def init_buffer_len(self):
        return len(self.buffer_init_action)

    def sample_init(self, batch_size):
        """
        为 REINFORCE 提取数据。
        注意：标准 REINFORCE 通常使用收集到的所有数据(Full Batch)。
        如果 batch_size 小于 buffer 大小，这里简单的返回前 batch_size 个数据，
        或者你可以修改为随机采样（但在 On-policy 中，全量使用效果最好）。
        """
        current_len = len(self.buffer_init_action)

        # 即使请求的 batch_size 小于当前数据量，在 REINFORCE 中我们也倾向于
        # 处理完所有收集到的数据然后清空。
        # 这里为了兼容接口，如果数据不够 batch_size 就不返回，够了就返回。
        if current_len < batch_size:
            return None  # 或者 raise Exception

        # 全量取出 (On-Policy: 使用当前策略产生的所有样本)
        indices = np.arange(current_len)

        # 构造 Batch
        # 将 list 转换为 numpy array，后续转 tensor 在 Policy 中进行
        batch = {
            'static_obs': np.array([self.buffer_static_obs[i] for i in indices]),
            'init_action': np.array([self.buffer_init_action[i] for i in indices]),
            'episode_reward': np.array([self.buffer_episode_reward[i] for i in indices]),
        }

        # 处理 additional_dict (log_prob, entropy 应该是 tensor)
        for key in self.buffer_init_additional_dict.keys():
            # 这里需要注意：如果存储的是 Tensor，堆叠最好用 torch.stack
            # 我们先取出列表，让外部 Policy 处理类型转换，或者在这里尝试堆叠
            raw_list = [self.buffer_init_additional_dict[key][i] for i in indices]

            if len(raw_list) > 0 and isinstance(raw_list[0], torch.Tensor):
                batch[key] = torch.stack(raw_list)
            else:
                batch[key] = np.array(raw_list)

        return batch

    def sample(self, batch_size):
        """
        Step-wise 采样 (Ego Agent 用)
        """
        prepared_ego_actions = []
        prepared_scenario_actions = []
        prepared_obs = []
        prepared_next_obs = []
        prepared_rewards = []
        prepared_dones = []

        # 将所有并行环境的数据展平
        for s_i in range(self.num_scenario):
            prepared_ego_actions += self.buffer_ego_actions[s_i]
            prepared_scenario_actions += self.buffer_scenario_actions[s_i]
            prepared_obs += self.buffer_obs[s_i]
            prepared_next_obs += self.buffer_next_obs[s_i]
            prepared_rewards += self.buffer_rewards[s_i]
            prepared_dones += self.buffer_dones[s_i]

        if len(prepared_rewards) < batch_size:
            return None

        sample_index = np.random.randint(0, len(prepared_rewards), size=batch_size)

        action = prepared_ego_actions if self.mode == 'train_agent' else prepared_scenario_actions
        batch = {
            'action': np.stack(action)[sample_index],
            'state': np.stack(prepared_obs)[sample_index],
            'n_state': np.stack(prepared_next_obs)[sample_index],
            'reward': np.stack(prepared_rewards)[sample_index],
            'done': np.stack(prepared_dones)[sample_index],
        }
        return batch


class PerceptionReplayBuffer:
    """
        This buffer supports parallel storing image states and labels for object detection
    """
    
    def __init__(self, num_scenario, mode, buffer_capacity=1000):
        self.mode = mode
        self.buffer_capacity = buffer_capacity
        self.num_scenario = num_scenario
        self.buffer_len = 0

        # buffers for different data type
        self.buffer_bbox_label = [[] for _ in range(num_scenario)]          # perception labels
        self.buffer_predictions = [[] for _ in range(num_scenario)]         # perception outputs
        self.buffer_scenario_actions = [[] for _ in range(num_scenario)]    # synthetic textures (attack)
        self.buffer_obs = [[] for _ in range(num_scenario)]                 # image observations (FPV observation)
        self.buffer_loss = [[] for _ in range(num_scenario)]                # object detection loss (IoU, class, etc.)
    
    def finish_one_episode(self):
        pass

    def reset_init_buffer(self):
        self.buffer_static_obs = []
        self.buffer_init_action = []
        self.buffer_episode_reward = []
        self.buffer_init_additional_dict = {}

    def store_init(self, data_list, additional_dict=None):
        pass
    
    def store(self, data_list, additional_dict=None):
        ego_actions = data_list[0]
        scenario_actions = data_list[1]
        obs = data_list[2]
        self.buffer_len += len(ego_actions)

        # separate trajectories according to infos
        for s_i in range(len(additional_dict)):
            sid = additional_dict[s_i]['s_id']
            self.buffer_predictions[sid].append(ego_actions[s_i]['od_result'])
            self.buffer_scenario_actions[sid].append(scenario_actions[s_i]['attack'])
            self.buffer_obs[sid].append(obs[s_i]['img'])
            self.buffer_bbox_label[sid].append(additional_dict[s_i]['bbox_label'])
            self.buffer_loss[sid].append(additional_dict[s_i]['iou_loss'])

    def sample(self, batch_size):
        # prepare concatenated list
        prepared_bbox_label = []
        prepared_predictions = []
        prepared_obs = []
        prepared_scenario_actions = []
        prepared_loss = []
        # get the length of each sub-buffer
        samples_per_trajectory = self.buffer_capacity // self.num_scenario # assume average over all sub-buffer
        for s_i in range(self.num_scenario):
            # select the latest samples starting from the end of buffer
            num_trajectory = len(self.buffer_loss[s_i])
            start_idx = np.max([0, num_trajectory - samples_per_trajectory])

            # concat
            prepared_bbox_label += self.buffer_bbox_label[s_i][start_idx:]
            prepared_predictions += self.buffer_predictions[s_i][start_idx:]
            prepared_scenario_actions += self.buffer_scenario_actions[s_i][start_idx:]
            prepared_obs += self.buffer_obs[s_i][start_idx:]
            prepared_loss += self.buffer_loss[s_i][start_idx:]
        # sample from concatenated list
        sample_index = np.random.randint(0, len(prepared_loss), size=batch_size)

        batch = {
            'label': np.stack(prepared_bbox_label)[sample_index, :],        
            # 'prediction': np.stack(prepared_predictions)[sample_index, :],     # TODO: Multiple/empty predictions should be stacked together
            # 'attack': np.stack(prepared_scenario_actions)[sample_index, :],
            # 'attack': torch.stack(prepared_scenario_actions)[sample_index, :],
            'image': np.stack(prepared_obs)[sample_index, :],
            'loss': np.stack(prepared_loss)[sample_index],                       # scalar with 1D 
        }
        
        return batch
