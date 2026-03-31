"""
td3_agent.py - Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent

实现基于 TD3 算法的连续动作控制，重点解决 DDPG 的 Q 值高估问题：
1. Clipped Double Q-learning: 使用两个 Critic，取最小值计算目标 Q 值。
2. Delayed Policy Updates: Actor 和目标网络的更新频率低于 Critic。
3. Target Policy Smoothing: 在目标动作上添加截断的噪声，平滑 Q 值估计。

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

class TD3Actor(nn.Module):
    """TD3 确定性 Actor Network"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()
        
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

class TD3Critic(nn.Module):
    """TD3 Double Q-Network Critic"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        # Q1 architecture
        layers1 = []
        prev_dim = state_dim + action_dim
        for h in hidden_dims:
            layers1.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU()
            ])
            prev_dim = h
        layers1.append(nn.Linear(prev_dim, 1))
        self.q1_net = nn.Sequential(*layers1)

        # Q2 architecture
        layers2 = []
        prev_dim = state_dim + action_dim
        for h in hidden_dims:
            layers2.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU()
            ])
            prev_dim = h
        layers2.append(nn.Linear(prev_dim, 1))
        self.q2_net = nn.Sequential(*layers2)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=1)
        return self.q1_net(xu), self.q2_net(xu)
        
    def Q1(self, state, action):
        xu = torch.cat([state, action], dim=1)
        return self.q1_net(xu)

class TD3Agent(BaseRLAgent, ContinuousActionMixin):
    """
    TD3 Agent for StateAwareNichePSO
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
                 exploration_noise: float = 0.2, # Tuned defaults for PSO
                 use_ou_noise: bool = True,      # Added OU Noise support
                 policy_noise: float = 0.3,      # Tuned defaults
                 noise_clip: float = 0.6,        # Tuned defaults
                 policy_delay: int = 3,          # Tuned defaults
                 device: str = 'cuda'):
        super().__init__()
        
        self.state_dim = state_dim
        self._action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise
        self.use_ou_noise = use_ou_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize Networks
        self.actor = TD3Actor(state_dim, action_dim, hidden_dims).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        
        self.critic = TD3Critic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Buffers & Noise
        self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim)
        if self.use_ou_noise:
            # Setting sigma higher to match the increased exploration_noise default
            self.ou_noise = OUNoise(action_dim, sigma=self.exploration_noise)
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
        """Update TD3 parameters"""
        if len(self.buffer) < self.batch_size:
            return {}
            
        total_critic_loss = 0
        total_actor_loss = 0
        
        for i in range(updates):
            state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
            state = state.to(self.device)
            action = action.to(self.device)
            reward = reward.to(self.device)
            next_state = next_state.to(self.device)
            done = done.to(self.device)
            
            # Normalize physical action back to [-1, 1] for Critic Evaluation
            act_tanh = (action - self.action_bias) / self.action_scale
            # Handle float inaccuracies
            act_tanh = torch.clamp(act_tanh, -0.999999, 0.999999) 
            
            # --- Critic Update ---
            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (
                    torch.randn_like(act_tanh) * self.policy_noise
                ).clamp(-self.noise_clip, self.noise_clip)
                
                next_action = (self.actor_target(next_state) + noise).clamp(-1.0, 1.0)
                
                # Compute the target Q value
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (1 - done) * self.gamma * target_Q

            # Get current Q estimates
            current_Q1, current_Q2 = self.critic(state, act_tanh)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            total_critic_loss += critic_loss.item()
            
            # --- Delayed Policy Updates ---
            self.update_count += 1
            if self.update_count % self.policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                total_actor_loss += actor_loss.item()

                # Soft update target models
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                    
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # We return the average loss over the updates. 
        # Note: Actor loss is only updated every policy_delay steps, so it might be 0 for some calls if updates < policy_delay.
        return {
            'critic_loss': total_critic_loss / updates,
            'actor_loss': total_actor_loss / max(1, updates // self.policy_delay),
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
        print(f"[TD3] Model saved to {path}")
        
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
            
        print(f"[TD3] Loaded from {path}")
        return agent

    def compute_policy_entropy(self, state: np.ndarray) -> float:
        """
        计算当前状态下策略分布的熵近似值 (nats)。
        TD3 为确定性策略，将探索噪声 + 目标策略平滑噪声 (policy_noise)
        综合建模为等效 Gaussian，计算解析熵。
        取 exploration_noise 和 policy_noise 的均均值作为有效 sigma。
        
        Args:
            state: 当前状态向量 (未使用，保持接口一致性)
        Returns:
            entropy: 策略分布熵近似值 (nats)
        """
        # TD3 两种噪声的有效标准差：取均就是计算总体不确定性
        sigma = (self.exploration_noise + self.policy_noise) / 2.0
        d = self._action_dim
        import math
        entropy = 0.5 * d * (1.0 + math.log(2.0 * math.pi * sigma ** 2 + 1e-8))
        return entropy
