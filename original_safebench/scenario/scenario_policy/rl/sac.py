import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from safebench.util.torch_util import CUDA, CPU, kaiming_init
from safebench.scenario.scenario_policy.base_policy import BasePolicy


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, action_dim)
        self.fc_log_std = nn.Linear(256, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.apply(kaiming_init)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        mu = self.tanh(self.fc_mu(x))
        log_std = torch.clamp(self.fc_log_std(x), -20, 2)  # 防止数值爆炸
        return mu, log_std


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.fc1 = nn.Linear(state_dim+action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.apply(kaiming_init)

    def forward(self, s, a):
        s = s.reshape(-1, self.state_dim)
        a = a.reshape(-1, self.action_dim)
        x = torch.cat((s, a), -1) # combination x and a
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SAC(BasePolicy):
    name = 'SAC'
    type = 'offpolicy'

    def __init__(self, config, logger):
        self.logger = logger
        self.policy_type = config['policy_type']

        self.buffer_start_training = config['buffer_start_training']
        self.lr = config['lr']
        self.continue_episode = config['continue_episode']
        self.state_dim = config['scenario_state_dim']
        self.action_dim = config['scenario_action_dim']
        self.min_Val = torch.tensor(config['min_Val']).float()
        self.batch_size = config['batch_size']
        self.update_iteration = config['update_iteration']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.alpha = config['alpha']  # 初始的温度参数

        # 自动温度调节
        # self.target_entropy = -float(self.action_dim)
        # self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True)
        # self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)

        self.model_id = config['model_id']
        self.model_path = os.path.join(config['ROOT_DIR'], config['model_path'])
        self.scenario_id = config['scenario_id']
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # create models
        self.actor = CUDA(Actor(self.state_dim, self.action_dim))
        self.q1 = CUDA(QNetwork(self.state_dim, self.action_dim))
        self.q2 = CUDA(QNetwork(self.state_dim, self.action_dim))
        self.target_q1 = CUDA(QNetwork(self.state_dim, self.action_dim))
        self.target_q2 = CUDA(QNetwork(self.state_dim, self.action_dim))

        # 复制参数到目标网
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        # create optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=self.lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=self.lr)

        self.mode = 'train'

    def set_mode(self, mode):
        self.mode = mode
        if mode == 'train':
            self.actor.train(), self.q1.train(), self.q2.train()
        elif mode == 'eval':
            self.actor.eval(), self.q1.eval(), self.q2.eval()
        else:
            raise ValueError(f'Unknown mode {mode}')

    def info_process(self, infos):
        info_batch = np.stack([i_i['actor_info'] for i_i in infos], axis=0)
        info_batch = info_batch.reshape(info_batch.shape[0], -1)
        return info_batch

    def get_init_action(self, state, deterministic=False):
        num_scenario = len(state)
        additional_info = {}
        return [None] * num_scenario, additional_info

    def get_action(self, state, infos, deterministic=False):
        state = self.info_process(infos)
        state = CUDA(torch.FloatTensor(state))
        mu, log_sigma = self.actor(state)

        if deterministic:
            action = mu
        else:
            sigma = torch.exp(log_sigma)
            dist = Normal(mu, sigma)
            z = dist.sample()
            action = torch.tanh(z)
        return CPU(action)

    def get_action_log_prob(self, state):
        mu, log_std = self.actor(state)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        z = dist.rsample()
        action = torch.tanh(z)
        eps = 1e-6
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + eps)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob

    def train(self, replay_buffer):
        if replay_buffer.buffer_len < self.buffer_start_training:
            return

        for _ in range(self.update_iteration):
            batch = replay_buffer.sample(self.batch_size)
            s = CUDA(torch.FloatTensor(batch['actor_info'])).reshape(self.batch_size, -1)
            s_next = CUDA(torch.FloatTensor(batch['n_actor_info'])).reshape(self.batch_size, -1)
            a = CUDA(torch.FloatTensor(batch['action']))
            r = CUDA(torch.FloatTensor(batch['reward'])).unsqueeze(-1)  # 对抗任务的奖励
            d = CUDA(torch.FloatTensor(1 - batch['done'])).unsqueeze(-1)

            # ------- 更新 Q 网络 --------
            with torch.no_grad():
                next_a, next_log_prob = self.get_action_log_prob(s_next)
                target_q1_val = self.target_q1(s_next, next_a)
                target_q2_val = self.target_q2(s_next, next_a)
                target_q_val = torch.min(target_q1_val, target_q2_val) - self.alpha * next_log_prob
                target_value = r + d * self.gamma * target_q_val

            q1_loss = F.smooth_l1_loss(self.q1(s, a), target_value)
            q2_loss = F.smooth_l1_loss(self.q2(s, a), target_value)

            self.q1_optimizer.zero_grad()
            q1_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q1.parameters(), 10.0)
            self.q1_optimizer.step()

            self.q2_optimizer.zero_grad()
            q2_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q2.parameters(), 10.0)
            self.q2_optimizer.step()

            # ------- 更新策略网络 --------
            new_a, log_prob = self.get_action_log_prob(s)
            q1_new = self.q1(s, new_a)
            q2_new = self.q2(s, new_a)
            q_new = torch.min(q1_new, q2_new)
            policy_loss = (self.alpha * log_prob - q_new).mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            # ------- 更新 α（温度系数） --------
            # alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            # self.alpha_optimizer.zero_grad()
            # alpha_loss.backward()
            # self.alpha_optimizer.step()
            # self.alpha = self.log_alpha.exp().item()

            # ------- 软更新目标网络 --------
            for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters()):
                target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)
            for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters()):
                target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)

    def save_model(self, episode):
        states = {
            'actor': self.actor.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'alpha': self.alpha
        }
        save_dir = os.path.join(self.model_path, str(self.scenario_id))
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f'model.sac.{self.model_id}.{episode:04}.torch')
        self.logger.log(f'>> Saving SAC model to {filepath}')
        with open(filepath, 'wb+') as f:
            torch.save(states, f)

    def load_model(self, scenario_configs=None):
        # 断点续训阶段加载
        if self.continue_episode > 0:
            model_filename = os.path.join(self.model_path, str(self.scenario_id), f'model.sac.{self.model_id}.{self.continue_episode-1:04}.torch')
            if os.path.exists(model_filename):
                self.logger.log(f'>> Loading {self.policy_type} model from {model_filename}')
                with open(model_filename, 'rb') as f:
                    checkpoint = torch.load(f)
                self.actor.load_state_dict(checkpoint['actor'])
                self.q1.load_state_dict(checkpoint['q1'])
                self.q2.load_state_dict(checkpoint['q2'])
                # self.alpha = checkpoint.get('alpha', self.alpha)
        # 评估阶段对应加载
        elif scenario_configs is not None:
            for config in scenario_configs:
                scenario_id = config.scenario_id
                model_file = config.parameters
                model_filename = os.path.join(self.model_path, str(scenario_id), model_file)
                if os.path.exists(model_filename):
                    self.logger.log(f'>> Loading {self.policy_type} model from {model_filename}')
                    with open(model_filename, 'rb') as f:
                        checkpoint = torch.load(f)
                    self.actor.load_state_dict(checkpoint['actor'])
                    self.q1.load_state_dict(checkpoint['q1'])
                    self.q2.load_state_dict(checkpoint['q2'])
                    # self.alpha = checkpoint.get('alpha', self.alpha)
        else:
            self.logger.log(f'>> Fail to find {self.policy_type} model, start from scratch', color='yellow')

