"""
ddpg_agent.py - Deep Deterministic Policy Gradient (DDPG) Agent

实现基于确定性策略梯度的 DDPG 算法：
1. Actor: 确定性策略输出，通过加噪进行探索。
2. Critic: 单个 Q-Network。
3. 探索: 采用 Ornstein-Uhlenbeck (OU) 过程添加时间序列相关的动作噪声。
4. Off-policy: 使用 Replay Buffer 进行经验回放训练。

适用于需要精细参数控制的小生境 PSO 搜索。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional
import copy
import os

from .base_agent import BaseRLAgent, ContinuousActionMixin, PSO_ACTION_BOUNDS
from .buffers import ReplayBuffer, OUNoise

class DDPGActor(nn.Module):
    """DDPG 确定性 Actor Network"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        # Shared layers
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
            
        self.base = nn.Sequential(*layers)
        
        # Head (Outputs Tanh action -> [-1, 1])
        self.mu_head = nn.Sequential(
            nn.Linear(prev_dim, action_dim),
            nn.Tanh()
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
                
    def forward(self, state):
        x = self.base(state)
        return self.mu_head(x)

class DDPGCritic(nn.Module):
    """DDPG Single Q-Network Critic"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        # Structure
        layers = []
        prev_dim = state_dim + action_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU()
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.q_net = nn.Sequential(*layers)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=1)
        return self.q_net(xu)

class DDPGAgent(BaseRLAgent, ContinuousActionMixin):
    """
    DDPG Agent for StateAwareNichePSO
    """
    ACTION_NAMES = ['w', 'c1', 'c2', 'velocity_scale']
    
    def __init__(self,
                 state_dim: int = 25,
                 action_dim: int = 4,
                 hidden_dims: List[int] = [128, 128, 64],
                 lr: float = 3e-4,
                 gamma: float = 0.7,
                 tau: float = 0.005,
                 batch_size: int = 256,
                 buffer_size: int = 100000,
                 exploration_noise: float = 0.1,  # 使用标准高斯噪声标准差（如果不适用OU）
                 use_ou_noise: bool = True,
                 device: str = 'cuda'):
        super().__init__()
        
        self.state_dim = state_dim
        self._action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise
        self.use_ou_noise = use_ou_noise
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize Networks
        self.actor = DDPGActor(state_dim, action_dim, hidden_dims).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        
        self.critic = DDPGCritic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Buffers & Noise
        self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim)
        if self.use_ou_noise:
            self.ou_noise = OUNoise(action_dim)
        else:
            self.ou_noise = None
            
        # Action Scaling Parameters based on PSO Parameter range
        self.action_low = torch.tensor([b[0] for b in PSO_ACTION_BOUNDS.values()], device=self.device)
        self.action_high = torch.tensor([b[1] for b in PSO_ACTION_BOUNDS.values()], device=self.device)
        self.action_scale = (self.action_high - self.action_low) / 2
        self.action_bias = (self.action_high + self.action_low) / 2
        
        self.update_count = 0

    @property
    def action_type(self) -> str:
        return 'continuous'
        
    @property
    def action_dim(self) -> int:
        return self._action_dim
        
    def reset_noise(self):
        if self.ou_noise is not None:
            self.ou_noise.reset()
            
    def choose_action(self, state: np.ndarray, deterministic: bool = False
                      ) -> Tuple[np.ndarray, Dict]:
        """
        Select action
        Returns:
            scaled_action: Actual PSO params
            info: {}
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_tanh = self.actor(state_tensor).squeeze(0).cpu().numpy()
            
        if not deterministic and not self.frozen:
            if self.use_ou_noise:
                noise = self.ou_noise.sample()
            else:
                noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
                
            action_tanh = action_tanh + noise
            action_tanh = np.clip(action_tanh, -1.0, 1.0)
            
        # Scale to PSO bounds
        action_tanh_tensor = torch.tensor(action_tanh, device=self.device)
        scaled_action = action_tanh_tensor * self.action_scale + self.action_bias
        
        return scaled_action.cpu().numpy(), {}
        
    def store_transition(self, state, action, reward, next_state, done, **kwargs):
        """
        Store transition.
         action is stored in the physical un-scaled form (e.g., [0.5, 1.2, 1.2, 0.9])
        """
        self.buffer.add(state, action, reward, next_state, done)
        
    def update(self, updates: int = 1):
        """Update DDPG parameters"""
        if len(self.buffer) < self.batch_size:
            return {}
            
        total_critic_loss = 0
        total_actor_loss = 0
        
        for _ in range(updates):
            state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
            state = state.to(self.device)
            action = action.to(self.device)
            reward = reward.to(self.device)
            next_state = next_state.to(self.device)
            done = done.to(self.device)
            
            # Normalize physical action back to [-1, 1] for Critic Evaluation
            action_tanh = (action - self.action_bias) / self.action_scale
            action_tanh = torch.clamp(action_tanh, -0.999999, 0.999999) 
            
            # --- Critic Update ---
            with torch.no_grad():
                next_action_tanh = self.actor_target(next_state)
                # target Q value based on deterministic next action
                target_q = self.critic_target(next_state, next_action_tanh)
                y = reward + (1 - done) * self.gamma * target_q
                
            current_q = self.critic(state, action_tanh)
            critic_loss = F.mse_loss(current_q, y)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # --- Actor Update ---
            # Compute actions from current policy
            policy_actions_tanh = self.actor(state)
            
            # Maximize Q value for policy actions, hence negative sign on objective
            actor_loss = -self.critic(state, policy_actions_tanh).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # --- Soft Update Target Networks ---
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            total_critic_loss += critic_loss.item()
            total_actor_loss += actor_loss.item()
            
        self.update_count += updates
        
        return {
            'critic_loss': total_critic_loss / updates,
            'actor_loss': total_actor_loss / updates,
        }

    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self._action_dim
            }
        }, path)
        print(f"[DDPG] Model saved to {path}")
        
    @classmethod
    def load(cls, path: str, freeze: bool = True):
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint.get('config', {})
        agent = cls(state_dim=config.get('state_dim', 25))
        
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.critic.load_state_dict(checkpoint['critic'])
        agent.actor_target.load_state_dict(checkpoint['actor'])
        agent.critic_target.load_state_dict(checkpoint['critic'])
            
        if freeze:
            agent.freeze()
            agent.actor.eval()
            
        print(f"[DDPG] Loaded from {path}")
        return agent

    def compute_policy_entropy(self, state: np.ndarray) -> float:
        """
        计算当前状态下策略分布的熵近似值 (nats)。
        DDPG 为确定性策略，将探索噪声 (OU/Gaussian) 建模为一个等效 Gaussian，
        计算解析熵 H = 0.5 * d * (1 + ln(2πσ²))。
        当训练早期探索噪声大，熵高；策略收敛后探索噪声不变但确定性部分变小。
        
        Args:
            state: 当前状态向量 (未使用，保持接口一致性)
        Returns:
            entropy: 策略分布熵近似值 (nats)
        """
        sigma = self.exploration_noise  # 直接用探索噪声标准差
        d = self._action_dim
        # Multivariate Gaussian 与生煮: H = 0.5*d*(1 + ln(2*pi*sigma^2))
        import math
        entropy = 0.5 * d * (1.0 + math.log(2.0 * math.pi * sigma ** 2 + 1e-8))
        return entropy
