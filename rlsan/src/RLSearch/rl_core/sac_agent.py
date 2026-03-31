"""
sac_agent.py - Soft Actor-Critic (SAC) Agent

实现了基于 Maximum Entropy RL 的 SAC 算法：
1. Actor: Tanh-Squashed Gaussian Policy (with Jacobian correction).
2. Critic: Double Q-Networks (mitigate overestimation).
3. Alpha: 自动熵调节 (Auto-tuned Entropy).
4. Off-policy: 使用 Replay Buffer 进行经验回放训练。

适用于需要精细参数控制的小生境 PSO 搜索。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional
import copy
import os

from .base_agent import BaseRLAgent, ContinuousActionMixin, PSO_ACTION_BOUNDS
from .buffers import ReplayBuffer

class SACActor(nn.Module):
    """SAC Actor Network (Tanh-Squashed Gaussian)"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int], 
                 log_std_min: float = -20, log_std_max: float = 2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
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
        
        # Heads
        self.mu_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
                
    def forward(self, state):
        x = self.base(state)
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
        
    def sample(self, state):
        mu, log_std = self(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        
        # Reparameterization trick
        u = dist.rsample()
        action = torch.tanh(u)
        
        # Jacobian correction for log_prob
        # log_prob(a) = log_prob(u) - sum(log(1 - tanh(u)^2))
        log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Calculate mean action for deterministic evaluation (squashed)
        mean_action = torch.tanh(mu)
        
        return action, log_prob, mean_action

class DoubleQCritic(nn.Module):
    """Double Q-Network"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        # Q1 architecture
        self.q1 = self._build_net(state_dim + action_dim, hidden_dims)
        
        # Q2 architecture
        self.q2 = self._build_net(state_dim + action_dim, hidden_dims)
        
        self._init_weights()
        
    def _build_net(self, input_dim, hidden_dims):
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU()
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, state, action):
        xu = torch.cat([state, action], dim=1)
        return self.q1(xu), self.q2(xu)

class SACAgent(BaseRLAgent, ContinuousActionMixin):
    """
    SAC Agent for StateAwareNichePSO
    """
    ACTION_NAMES = ['w', 'c1', 'c2', 'velocity_scale']
    
    def __init__(self,
                 state_dim: int = 25,
                 action_dim: int = 4,
                 hidden_dims: List[int] = [256, 256],
                 lr: float = 3e-4,
                 gamma: float = 0.7,
                 tau: float = 0.005,
                 alpha: float = 0.2, # Initial temperature
                 auto_entropy_tuning: bool = True,
                 batch_size: int = 256,
                 buffer_size: int = 100000,
                 device: str = 'cuda'):
        super().__init__()
        
        self.state_dim = state_dim
        self._action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Initialize Networks
        self.actor = SACActor(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic = DoubleQCritic(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # Auto Entropy Tuning
        self.auto_entropy_tuning = auto_entropy_tuning
        if self.auto_entropy_tuning:
            self.target_entropy = -float(action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = torch.tensor(alpha, device=self.device)
            
        # Replay Buffer
        self.buffer = ReplayBuffer(buffer_size, state_dim, action_dim)
        
        # Action Scaling
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
            action, _, mean = self.actor.sample(state_tensor)
            
        if deterministic:
            action_tanh = mean
        else:
            action_tanh = action
            
        # Scale to PSO bounds
        scaled_action = action_tanh * self.action_scale + self.action_bias
        return scaled_action.cpu().numpy()[0], {}
        
    def store_transition(self, state, action, reward, next_state, done, **kwargs):
        """
        Store transition.
        Note: action stored should be the UN-SCALED (i.e. physical) action, 
        SAC logic handles the reverse scaling or we store raw tanh action?
        Standard practice: Store the physical action that interacted with env.
        In update(), we must Normalize it back to [-1, 1] for the network.
        """
        self.buffer.add(state, action, reward, next_state, done)
        
    def update(self, updates: int = 1):
        """Update SAC parameters"""
        if len(self.buffer) < self.batch_size:
            return {}
            
        total_critic_loss = 0
        total_actor_loss = 0
        total_alpha_loss = 0
        total_alpha = 0
        
        for _ in range(updates):
            state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
            state = state.to(self.device)
            action = action.to(self.device)
            # PPO is using raw rewards. For SAC, rewards can sometimes be very large.
            # Using clipping or tanh for rewards here is redundant if PSO already scales them, but just in case:
            reward = reward.to(self.device)
            next_state = next_state.to(self.device)
            done = done.to(self.device)
            
            # Normalize action back to [-1, 1] for Q-network
            # Physical Action -> Tanh Space
            action_tanh = (action - self.action_bias) / self.action_scale
            # Clamp to prevent numerical errors 
            action_tanh = torch.clamp(action_tanh, -0.999999, 0.999999) 
            
            # --- Critic Update ---
            with torch.no_grad():
                next_action, next_log_prob, _ = self.actor.sample(next_state)
                q1_next, q2_next = self.critic_target(next_state, next_action)
                min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
                q_target = reward + (1 - done) * self.gamma * min_q_next
                
            q1, q2 = self.critic(state, action_tanh)
            critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # --- Actor Update ---
            new_action, log_prob, _ = self.actor.sample(state)
            q1_new, q2_new = self.critic(state, new_action)
            min_q_new = torch.min(q1_new, q2_new)
            
            actor_loss = (self.alpha * log_prob - min_q_new).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # --- Alpha Update ---
            if self.auto_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                self.alpha = self.log_alpha.exp()
                total_alpha_loss += alpha_loss.item()
            
            # --- Soft Update Target ---
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            total_critic_loss += critic_loss.item()
            total_actor_loss += actor_loss.item()
            total_alpha += self.alpha.item()
            
        self.update_count += updates
        
        return {
            'critic_loss': total_critic_loss / updates,
            'actor_loss': total_actor_loss / updates,
            'alpha_loss': total_alpha_loss / updates,
            'alpha': total_alpha / updates
        }

    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy_tuning else None,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self._action_dim
            }
        }, path)
        print(f"[SAC] Model saved to {path}")
        
    @classmethod
    def load(cls, path: str, freeze: bool = True):
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint.get('config', {})
        agent = cls(state_dim=config.get('state_dim', 25))
        
        agent.actor.load_state_dict(checkpoint['actor'])
        agent.critic.load_state_dict(checkpoint['critic'])
        agent.critic_target.load_state_dict(checkpoint['critic'])
        
        if checkpoint['log_alpha'] is not None and agent.auto_entropy_tuning:
            agent.log_alpha.data.copy_(checkpoint['log_alpha'])
            agent.alpha = agent.log_alpha.exp()
            
        if freeze:
            agent.freeze()
            agent.actor.eval()
            
        print(f"[SAC] Loaded from {path}")
        return agent

    def compute_policy_entropy(self, state: np.ndarray) -> float:
        """
        计算当前状态下策略分布的熵 (nats)。
        SAC 使用 Tanh-Squashed Gaussian，熵近似为 -E[log_prob]。
        通过单次采样估计。
        
        Args:
            state: 当前状态向量
        Returns:
            entropy: 策略分布熵 (nats)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, log_prob, _ = self.actor.sample(state_tensor)
            # log_prob shape: (1, 1), entropy = -log_prob
            entropy = -log_prob.mean().item()
        return entropy
