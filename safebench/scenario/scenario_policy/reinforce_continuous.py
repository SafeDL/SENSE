import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from safebench.scenario.scenario_policy.base_policy import BasePolicy
from safebench.util.torch_util import CUDA, CPU


def normalize_routes(routes):
    mean_x = np.mean(routes[:, 0:1])
    max_x = np.max(np.abs(routes[:, 0:1]))
    x_1_2 = (routes[:, 0:1] - mean_x) / max_x

    mean_y = np.mean(routes[:, 1:2])
    max_y = np.max(np.abs(routes[:, 1:2]))
    y_1_2 = (routes[:, 1:2] - mean_y) / max_y

    route = np.concatenate([x_1_2, y_1_2], axis=0)
    return route


class AutoregressiveModel(nn.Module):
    def __init__(self, num_waypoint=30, standard_action_dim=False):
        super(AutoregressiveModel, self).__init__()
        self.standard_action_dim = standard_action_dim  # 标准化动作输出
        input_size = num_waypoint*2 + 1
        hidden_size_1 = 32

        self.a_os = 1
        self.b_os = 1
        self.c_os = 1
        if self.standard_action_dim:
            self.d_os = 1

        self.relu = nn.ReLU()
        self.fc_input = nn.Sequential(nn.Linear(input_size, hidden_size_1))
        self.fc_action_a = nn.Sequential(nn.Linear(hidden_size_1, self.a_os*2))
        self.fc_action_b = nn.Sequential(nn.Linear(1+hidden_size_1, self.b_os*2))
        self.fc_action_c = nn.Sequential(nn.Linear(1+1+hidden_size_1, self.c_os*2))
        if self.standard_action_dim:
            self.fc_action_d = nn.Sequential(nn.Linear(1+1+1+hidden_size_1, self.d_os*2))

    def sample_action(self, normal_action, action_os):
        # get the mu and sigma
        # action_os: 动作分支输出的物理维度 (output size),它控制了网络输出多少 μ/σ,用来采样该动作对应的高斯分布。
        mu = normal_action[:, :action_os]
        sigma = F.softplus(normal_action[:, action_os:])

        # calculate the probability by mu and sigma of normal distribution
        eps = CUDA(torch.randn(mu.size()))
        action = mu + sigma * eps
        return action, mu, sigma

    def forward(self, x, determinstic):
        # p(s)
        s = self.fc_input(x)
        s = self.relu(s)

        # p(a|s)
        normal_a = self.fc_action_a(s)
        action_a, mu_a, sigma_a = self.sample_action(normal_a, self.a_os)

        # p(b|a,s)
        state_sample_a = torch.cat((s, mu_a), dim=1) if determinstic else torch.cat((s, action_a), dim=1) 
        normal_b = self.fc_action_b(state_sample_a)
        action_b, mu_b, sigma_b = self.sample_action(normal_b, self.b_os)

        # p(c|a,b,s)
        state_sample_a_b = torch.cat((s, mu_a, mu_b), dim=1) if determinstic else torch.cat((s, action_a, action_b), dim=1)
        normal_c = self.fc_action_c(state_sample_a_b)
        action_c, mu_c, sigma_c = self.sample_action(normal_c, self.c_os)

        # p(d|a,b,c,s)
        if self.standard_action_dim:
            state_sample_a_b_c = torch.cat((s, mu_a, mu_b, mu_c), dim=1) if determinstic else torch.cat((s, action_a, action_b, action_c), dim=1)
            normal_d = self.fc_action_d(state_sample_a_b_c)
            action_d, mu_d, sigma_d = self.sample_action(normal_d, self.d_os)

        # concate
        if self.standard_action_dim:
            action = torch.cat((action_a, action_b, action_c, action_d), dim=1) # [B, 4]
            mu = torch.cat((mu_a, mu_b, mu_c, mu_d), dim=1)                     # [B, 4]
            sigma = torch.cat((sigma_a, sigma_b, sigma_c, sigma_d), dim=1)      # [B, 4]
        else:
            action = torch.cat((action_a, action_b, action_c), dim=1)           # [B, 3]
            mu = torch.cat((mu_a, mu_b, mu_c), dim=1)                           # [B, 3]
            sigma = torch.cat((sigma_a, sigma_b, sigma_c), dim=1)               # [B, 3]

        # 对动作进行tanh压缩到[-1, 1]范围内
        action = torch.tanh(action)
        return mu, sigma, action


class REINFORCE(BasePolicy):
    name = 'reinforce'
    type = 'init_state'

    def __init__(self, scenario_config, logger):
        self.logger = logger
        self.num_waypoint = 30
        self.continue_episode = 0
        self.num_scenario = scenario_config['num_scenario']
        self.batch_size = scenario_config['batch_size']
        self.model_path = os.path.join(scenario_config['ROOT_DIR'], scenario_config['model_path'])
        self.model_id = scenario_config['model_id']
        self.lr = scenario_config['lr']
        self.entropy_weight = 0.001
        self.standard_action_dim = scenario_config["standard_action_dim"]
        self.model = CUDA(AutoregressiveModel(self.num_waypoint, self.standard_action_dim))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # 价值基线（用于上下文 bandit 的 V(s)）
        value_hidden = 64
        self.value_net = CUDA(nn.Sequential(
            nn.Linear(self.num_waypoint * 2 + 1, value_hidden),
            nn.ReLU(),
            nn.Linear(value_hidden, 1)
        ))
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)

    def train(self, replay_buffer):
        """引入状态价值基线，使用优势训练"""
        if replay_buffer.init_buffer_len < self.batch_size:
            return

        # get episode reward
        batch = replay_buffer.sample_init(self.batch_size)
        episode_reward = batch['episode_reward']
        log_prob = batch['log_prob']
        entropy = batch['entropy']
        # 用于价值基线的状态表示
        state_embed = batch.get('state_embed', None)

        episode_reward = CUDA(torch.tensor(episode_reward, dtype=torch.float32))
        if state_embed is not None:
            with torch.no_grad():
                v_pred = self.value_net(state_embed).squeeze(1)
            advantage = episode_reward - v_pred
        else:
            mean_r = episode_reward.mean()
            std_r = episode_reward.std(unbiased=False) + 1e-8
            advantage = (episode_reward - mean_r) / std_r

        advantage = torch.clamp(advantage, -50, 50)

        # policy loss（maximize E[log_pi * A]）
        policy_loss = (-log_prob * advantage - self.entropy_weight * entropy).mean(dim=0)
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # value loss（MSE）
        if state_embed is not None:
            v_value = self.value_net(state_embed).squeeze(1)
            value_loss = F.mse_loss(v_value, episode_reward)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            self.logger.log('>> Training loss: policy {:.4f} | value loss {:.4f}'.format(policy_loss.item(), value_loss.item()))
        else:
            self.logger.log('>> Training loss: policy {:.4f}'.format(policy_loss.item()))

        replay_buffer.reset_init_buffer()

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.model.train()
        elif mode == 'eval':
            self.model.eval()
        else:
            raise ValueError(f'Unknown mode {mode}')

    def proceess_init_state(self, state):
        processed_state_list = []
        for s_i in range(len(state)):
            route = state[s_i]['route']
            target_speed = state[s_i]['target_speed'] / 36.0  # 相当于从km/h变成m/s再速度除以10进行标准化

            index = np.linspace(1, len(route) - 1, self.num_waypoint).tolist()
            index = [int(i) for i in index]
            route_norm = normalize_routes(route[index])[:, 0] # [num_waypoint*2]
            processed_state = np.concatenate((route_norm, [target_speed]), axis=0).astype('float32')
            processed_state_list.append(processed_state)
        
        processed_state_list = np.stack(processed_state_list, axis=0)
        return processed_state_list

    def get_action(self, state, infos, deterministic=False):
        return [None] * self.num_scenario

    def get_init_action(self, state, deterministic=True):
        # the state should be a sequence of route waypoints
        processed_state = self.proceess_init_state(state)
        processed_state = CUDA(torch.from_numpy(processed_state))

        # Model outputs parameters for pre-squash Gaussian; we will tanh-squash
        mu, sigma, a = self.model.forward(processed_state, deterministic)
        sigma = torch.clamp(sigma, 1e-3, 2.0)
        action_dist = Normal(mu, sigma)

        # log_prob = action_dist.log_prob(a).sum(dim=1) # [B]

        # change-of-variables correction: log_prob(u) - sum log(1 - tanh(u)^2)
        u = mu + sigma * 1e-6  # pre-tanh
        log_prob_u = action_dist.log_prob(u).sum(dim=1)
        log_prob_correction = torch.log(1 - a.pow(2) + 1e-6).sum(dim=1)
        log_prob = log_prob_u - log_prob_correction

        action_entropy = 0.5 * (torch.log(2 * torch.tensor(np.pi, device=sigma.device)) + 1.0 + 2 * torch.log(sigma))
        entropy = action_entropy.sum(dim=1)

        action = CPU(a)
        additional_info = {'log_prob': log_prob, 'entropy': entropy, 'state_embed': processed_state}
        return action, additional_info

    # start from scratch 训练自己的权重
    def load_model(self, scenario_configs=None):
        self.model = CUDA(AutoregressiveModel(self.num_waypoint, self.standard_action_dim))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # 加载的权重形式要匹配为最后一次断点训练的样式, eg: model_path/1.pt
        model_filename = os.path.join(self.model_path, f'{self.model_id}.pt')
        if os.path.exists(model_filename):
            self.logger.log(f'>> Loading lc model from {model_filename}')
            with open(model_filename, 'rb') as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint['parameters'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            # 价值网络可选加载
            if 'value_parameters' in checkpoint and 'value_optimizer' in checkpoint:
                self.value_net.load_state_dict(checkpoint['value_parameters'])
                self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        else:
            self.logger.log(f'>> Fail to find lc model from {model_filename}', color='yellow')

    def save_model(self, epoch):
        if not os.path.exists(self.model_path):
            self.logger.log(f'>> Creating folder for saving model: {self.model_path}')
            os.makedirs(self.model_path)
        model_filename = os.path.join(self.model_path, f'{self.model_id}_{epoch}.pt')
        self.logger.log(f'>> Saving lc model to {model_filename}')
        with open(model_filename, 'wb+') as f:
            torch.save({
                'parameters': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'value_parameters': self.value_net.state_dict(),
                'value_optimizer': self.value_optimizer.state_dict(),
            }, f)

