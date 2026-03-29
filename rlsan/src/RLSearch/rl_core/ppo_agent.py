"""
ppo_agent.py - Proximal Policy Optimization (PPO) Agent

直接输出PSO参数 (w, c1, c2, velocity_scale)，使用Gaussian分布采样。
相比离散PPO可以实现更精细的参数控制。

特点：
- Actor输出动作均值和标准差
- 使用tanh压缩到有效参数范围
- 支持SAC风格的可学习标准差或固定标准差
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Tuple, List, Dict, Optional

from .base_agent import BaseRLAgent, ContinuousActionMixin, PSO_ACTION_BOUNDS


class ContinuousActorCritic(nn.Module):
    """连续动作Actor-Critic网络"""
    
    # 动作参数定义
    ACTION_NAMES = ['w', 'c1', 'c2', 'velocity_scale']
    ACTION_DIM = 4
    
    def __init__(self, 
                 state_dim: int,
                 hidden_dims: List[int] = [128, 128, 64],
                 log_std_init: float = 0.0,  # 增大初始值，鼓励探索
                 log_std_min: float = -1.5,  # 提高下限，避免过早收敛
                 log_std_max: float = 1.0):  # 允许更大探索
        super().__init__()
        
        self.action_dim = self.ACTION_DIM
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # 动作范围
        self.action_low = torch.tensor([
            PSO_ACTION_BOUNDS['w'][0],
            PSO_ACTION_BOUNDS['c1'][0],
            PSO_ACTION_BOUNDS['c2'][0],
            PSO_ACTION_BOUNDS['velocity_scale'][0],
        ], dtype=torch.float32)
        
        self.action_high = torch.tensor([
            PSO_ACTION_BOUNDS['w'][1],
            PSO_ACTION_BOUNDS['c1'][1],
            PSO_ACTION_BOUNDS['c2'][1],
            PSO_ACTION_BOUNDS['velocity_scale'][1],
        ], dtype=torch.float32)
        
        self.action_scale = (self.action_high - self.action_low) / 2
        self.action_bias = (self.action_high + self.action_low) / 2
        
        # 共享特征提取层
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        self.shared = nn.Sequential(*layers)
        
        # Actor头（输出动作均值 - 无Tanh，允许高斯分布均值在任意范围）
        self.actor_mean = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], self.action_dim),
            # nn.Tanh()  # REMOVED: Tanh is applied after sampling
        )
        
        # 可学习的log标准差
        self.log_std = nn.Parameter(torch.ones(self.action_dim) * log_std_init)
        
        # Critic头（输出状态价值）
        self.critic = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def to(self, device):
        super().to(device)
        self.action_low = self.action_low.to(device)
        self.action_high = self.action_high.to(device)
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return self
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        features = self.shared(state)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return action_mean, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取动作 (Tanh-Squashed Gaussian)
        """
        action_mean, value = self.forward(state)
        
        # 限制log_std范围
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        dist = Normal(action_mean, std)
        
        if deterministic:
            # Deterministic: 直接使用均值并通过tanh
            action_unsquashed = action_mean
        else:
            # Re-parameterization sampling
            action_unsquashed = dist.rsample()
            
        # Tanh Squashing
        action_tanh = torch.tanh(action_unsquashed)
        
        # 缩放到实际参数范围
        action = action_tanh * self.action_scale + self.action_bias
        
        # 计算 Log Prob (带 Jacobian Correction)
        # log_prob(y) = log_prob(x) - log(det|dy/dx|)
        # dy/dx = 1 - tanh^2(x)
        # log(1 - tanh^2(x))
        log_prob = dist.log_prob(action_unsquashed)
        log_prob -= torch.log(self.action_scale * (1 - action_tanh.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        
        return action, log_prob, value.squeeze(-1), action_tanh
    
    def evaluate(self, states: torch.Tensor, actions: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        评估动作 (用于PPO更新)
        """
        action_mean, values = self.forward(states)
        
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        dist = Normal(action_mean, std)
        
        # 1. 反变换: Action Space -> Tanh Space -> Unsquashed Space
        # Normalize: [low, high] -> [-1, 1]
        action_tanh = (actions - self.action_bias) / self.action_scale
        
        # Clip to avoid NaNs in atanh (numerical stability)
        action_tanh = torch.clamp(action_tanh, -0.999999, 0.999999)
        
        # Inverse Tanh: [-1, 1] -> (-inf, inf)
        action_unsquashed = torch.atanh(action_tanh)
        
        # 2. 计算 Log Prob (带 Jacobian Correction)
        log_prob = dist.log_prob(action_unsquashed)
        log_prob -= torch.log(self.action_scale * (1 - action_tanh.pow(2)) + 1e-6)
        log_probs = log_prob.sum(dim=-1)
        
        # 3. 计算 Entropy
        # 注意: Tanh变换后的熵没有简单的解析解。
        # 这里我们近似使用高斯分布的熵，这在PPO中通常足够有效且稳定。
        # 严格来说应该减去 E[log(Jacobian)]，但计算复杂。
        entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, values.squeeze(-1), entropy


class PPOBuffer:
    """PPO经验缓冲区"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
    
    def get(self) -> Tuple:
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.log_probs),
            np.array(self.rewards),
            np.array(self.values),
            np.array(self.dones),
        )
    
    def __len__(self):
        return len(self.states)


class PPOAgent(BaseRLAgent, ContinuousActionMixin):
    """
    PPO Agent for StateAwareNichePSO
    
    直接输出PSO参数 (w, c1, c2, velocity_scale)
    """
    
    ACTION_NAMES = ContinuousActorCritic.ACTION_NAMES
    
    def __init__(self,
                 state_dim: int = 25,
                 lr: float = 3e-4,
                 gamma: float = 0.7,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.005,  # 降低熵系数，避免策略过于随机
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 10,
                 mini_batch_size: int = 64,
                 hidden_dims: List[int] = [128, 128, 64],
                 device: str = 'cuda'):
        
        super().__init__()

        self.state_dim = state_dim
        self._action_dim = 4  # w, c1, c2, velocity_scale
        self.hidden_dims = hidden_dims  # ← 保存 hidden_dims
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 网络
        self.network = ContinuousActorCritic(
            state_dim, hidden_dims
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # 缓冲区
        self.buffer = PPOBuffer()
        
        # 统计
        self.update_count = 0
        
        # 存储最后的log_prob和value用于store_transition
        self._last_log_prob = None
        self._last_value = None
    
    @property
    def action_type(self) -> str:
        return 'continuous'
    
    @property
    def action_dim(self) -> int:
        return self._action_dim
    
    @property
    def action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return (
            self.network.action_low.cpu().numpy(),
            self.network.action_high.cpu().numpy()
        )
    
    def choose_action(self, state: np.ndarray, deterministic: bool = False
                      ) -> Tuple[np.ndarray, Dict]:
        """
        选择动作
        
        Returns:
            action: PSO参数数组 [w, c1, c2, velocity_scale]
            info: 包含log_prob, value, mean的字典
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value, mean = self.network.get_action(
                state_tensor, deterministic
            )
        
        action_np = action.squeeze(0).cpu().numpy()
        
        # 存储用于后续store_transition
        self._last_log_prob = log_prob.item()
        self._last_value = value.item()
        
        info = {
            'log_prob': self._last_log_prob,
            'value': self._last_value,
            'mean': mean.squeeze(0).cpu().numpy(),
        }
        
        return action_np, info
    
    def store_transition(self, state: np.ndarray, action: np.ndarray,
                        reward: float, next_state: np.ndarray,
                        done: bool = False,
                        log_prob: float = None,
                        value: float = None):
        """
        存储经验
        
        Args:
            state: 状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
            log_prob: 对数概率（可选，优先使用传入值）
            value: 价值估计（可选，优先使用传入值）
        """
        # 优先使用传入的参数，否则使用最后缓存的值
        if log_prob is None:
            log_prob = self._last_log_prob if self._last_log_prob is not None else 0.0
        if value is None:
            value = self._last_value if self._last_value is not None else 0.0
        
        self.buffer.store(state, action, log_prob, reward, value, done)
    
    def compute_gae(self, rewards, values, dones, next_value):
        """计算GAE"""
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def update(self, next_value: float = 0.0) -> Dict[str, float]:
        """PPO更新"""
        if len(self.buffer) < self.mini_batch_size:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
        
        states, actions, old_log_probs, rewards, values, dones = self.buffer.get()
        
        # 计算GAE
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # 归一化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # PPO多轮更新
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        update_steps = 0
        
        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # 评估当前策略
                new_log_probs, new_values, entropy = self.network.evaluate(
                    batch_states, batch_actions
                )
                
                # PPO Clipped Objective
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 
                                   1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(new_values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.value_coef * value_loss + 
                       self.entropy_coef * entropy_loss)
                
                # 更新
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                update_steps += 1
        
        self.buffer.clear()
        self.update_count += 1
        
        return {
            'policy_loss': total_policy_loss / max(update_steps, 1),
            'value_loss': total_value_loss / max(update_steps, 1),
            'entropy': total_entropy / max(update_steps, 1),
        }
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self._action_dim,
                'hidden_dims': self.hidden_dims,
                'gamma': self.gamma,
                'clip_epsilon': self.clip_epsilon,
            }
        }, path)
        print(f"[PPO] Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, freeze: bool = True) -> 'PPOAgent':
        """加载模型"""
        checkpoint = torch.load(path, map_location='cpu')

        config = checkpoint.get('config', {})
        agent = cls(
            state_dim=config.get('state_dim', 25),
            hidden_dims=config.get('hidden_dims', [128, 128, 64]),
        )

        agent.network.load_state_dict(checkpoint['network'])
        if 'optimizer' in checkpoint:
            try:
                agent.optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                pass
        agent.update_count = checkpoint.get('update_count', 0)
        
        if freeze:
            agent.freeze()
            agent.network.eval()
        
        print(f"[PPO] Loaded from {path} (freeze={freeze})")
        return agent
    
    def get_action_dict(self, action: np.ndarray) -> Dict[str, float]:
        """将动作数组转换为参数字典"""
        return {
            'w': float(action[0]),
            'c1': float(action[1]),
            'c2': float(action[2]),
            'velocity_scale': float(action[3]),
        }


# ===== 测试代码 =====
if __name__ == "__main__":
    print("Testing PPOAgent...")
    
    agent = PPOAgent(state_dim=25)
    
    # 模拟状态
    state = np.random.randn(25).astype(np.float32)
    
    # 测试动作选择
    action, info = agent.choose_action(state)
    print(f"Action: {action}")
    print(f"Action dict: {agent.get_action_dict(action)}")
    print(f"Info: {info}")
    
    # 测试存储和更新
    for _ in range(100):
        action, info = agent.choose_action(state)
        agent.store_transition(state, action, 0.5, state, False)
    
    metrics = agent.update()
    print(f"Update metrics: {metrics}")
    
    print("Test passed!")
