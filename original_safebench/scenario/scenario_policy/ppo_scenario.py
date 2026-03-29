"""
基于PPO的场景初始状态生成策略
"""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import torch
from torch.distributions import Normal
from safebench.scenario.scenario_policy.base_policy import BasePolicy
from safebench.util.torch_util import CUDA, CPU
from copy import deepcopy


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
    def __init__(self, num_waypoint=30, standard_action_dim=True):
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


class PPO(BasePolicy):
    name = 'ppo'
    type = 'init_state'

    def __init__(self, scenario_config, logger):
        self.logger = logger
        self.num_waypoint = 30
        self.continue_episode = 0
        self.num_scenario = scenario_config['num_scenario']
        self.batch_size = scenario_config['batch_size']
        self.train_iteration = scenario_config.get('train_iteration', 4)
        self.gamma = scenario_config.get('gamma', 0.99)
        self.clip_epsilon = scenario_config.get('clip_epsilon', 0.2)
        self.entropy_weight = scenario_config.get('entropy_weight', 0.01)

        self.model_path = os.path.join(scenario_config['ROOT_DIR'], scenario_config['model_path'])
        self.model_id = scenario_config['model_id']
        self.lr = scenario_config['lr']
        self.standard_action_dim = True

        self.policy = CUDA(AutoregressiveModel(self.num_waypoint, self.standard_action_dim))
        self.old_policy = deepcopy(self.policy)  # 旧策略，用于 ratio 计算
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        # 价值网络
        value_hidden = 64
        self.value_net = CUDA(torch.nn.Sequential(
            torch.nn.Linear(self.num_waypoint * 2 + 1, value_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(value_hidden, 1)
        ))
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)

        self.best_reward = -float('inf')
        self.best_model_file = os.path.join(self.model_path, f'{self.model_id}_best.pt')

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.policy.train()
            self.value_net.train()
        elif mode == 'eval':
            self.policy.eval()
            self.value_net.eval()

    def proceess_init_state(self, state):
        processed_state_list = []
        for s_i in range(len(state)):
            route = state[s_i]['route']
            target_speed = state[s_i]['target_speed'] / 36.0
            index = np.linspace(1, len(route) - 1, self.num_waypoint).tolist()
            index = [int(i) for i in index]
            route_norm = normalize_routes(route[index])[:, 0]
            processed_state = np.concatenate((route_norm, [target_speed]), axis=0).astype('float32')
            processed_state_list.append(processed_state)
        processed_state_list = np.stack(processed_state_list, axis=0)
        return processed_state_list

    def get_action(self, state, infos, deterministic=False):
        return [None] * self.num_scenario

    def get_init_action(self, state, deterministic=False):
        processed_state = self.proceess_init_state(state)
        processed_state = CUDA(torch.from_numpy(processed_state))
        mu, sigma, a = self.policy.forward(processed_state, deterministic)
        action_dist = Normal(mu, sigma)
        entropy = action_dist.entropy().sum(dim=1)
        action = CPU(a)
        return action, {'state_embed': processed_state, 'action': a, 'entropy': entropy}

    def train(self, replay_buffer):
        if len(replay_buffer.buffer_episode_reward) + len(replay_buffer.high_buffer_episode_reward) < self.batch_size:
            return

        # 从 buffer 取样
        batch = replay_buffer.sample_init(self.batch_size)

        states = batch['state_embed'].detach().clone()
        actions = batch['action'].detach().clone()
        rewards = torch.tensor(batch['episode_reward'], dtype=torch.float32)
        entropy = batch['entropy'].detach().clone()

        states = CUDA(states)
        actions = CUDA(actions)
        rewards = CUDA(torch.tensor(rewards, dtype=torch.float32))

        # 更新旧策略
        self.old_policy.load_state_dict(self.policy.state_dict())

        # 优势计算
        with torch.no_grad():
            v_pred = self.value_net(states).squeeze(1)
            advantages = rewards - v_pred

        # 多次迭代 PPO 更新
        for it in range(self.train_iteration):
            mu, sigma, _ = self.policy.forward(states, determinstic=False)
            dist = Normal(mu, sigma)
            new_log_prob = dist.log_prob(actions).sum(dim=1)
            old_mu, old_sigma, _ = self.old_policy.forward(states, determinstic=False)
            old_dist = Normal(old_mu, old_sigma)
            old_log_prob = old_dist.log_prob(actions).sum(dim=1)

            ratio = torch.exp(new_log_prob - old_log_prob)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_weight * entropy.mean()

            # 更新策略网络
            self.optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 更新价值网络
            value_pred = self.value_net(states.detach()).squeeze(1)
            value_loss = F.mse_loss(value_pred, rewards)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            self.logger.log('>> Training loss: policy {:.4f} | value loss {:.4f}'.format(policy_loss.item(), value_loss.item()))

        avg_reward = rewards.mean().item()
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            self.logger.log(f'>> New best reward {avg_reward:.4f}, saving best model...')
            torch.save({
                'parameters': self.policy.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'value_parameters': self.value_net.state_dict(),
                'value_optimizer': self.value_optimizer.state_dict(),
            }, self.best_model_file)

        replay_buffer.reset_init_buffer()

    def load_model(self, scenario_configs=None):
        model_filename = os.path.join(self.model_path, f'{self.model_id}.pt')
        if os.path.exists(model_filename):
            self.logger.log(f">> Load model from {model_filename}", color='green')
            checkpoint = torch.load(model_filename)
            self.policy.load_state_dict(checkpoint['parameters'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.value_net.load_state_dict(checkpoint['value_parameters'])
            self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        else:
            self.logger.log(f'>> Fail to find model from {model_filename}, start from scratch', color='yellow')


    def save_model(self, epoch):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        model_filename = os.path.join(self.model_path, f'{self.model_id}.pt')
        torch.save({
            'parameters': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'value_parameters': self.value_net.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
        }, model_filename)
