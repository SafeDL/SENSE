"""
trpo_agent.py - Trust Region Policy Optimization (TRPO) Agent

实现基于自然策略梯度的 TRPO 算法：
1. Actor (Policy): 高斯策略网络输出动作分布的均值和对数标准差。
2. Critic (Value): V-Network 评估状态价值。
3. 更新方式: On-policy 收集轨迹，通过 Generalized Advantage Estimation (GAE) 计算优势函数。
4. 优化核心: 
   - 使用共轭梯度法 (Conjugate Gradient) 估算海森矩阵的逆与策略梯度的乘积 (Fisher Vector Product)。
   - 使用线搜索 (Line Search) 在满足 KL 散度约束下更新策略参数，保证单调递增。

适用于与基于PPO的同一套小生境 PSO 任务实现公平对照。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List, Dict
import copy
from .base_agent import BaseRLAgent, ContinuousActionMixin, PSO_ACTION_BOUNDS

class ExperienceBuffer:
    """与 PPO 完全相同的 On-Policy 经验缓冲区"""
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.ptr = 0
        
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
    def add(self, state, action, reward, value, log_prob, done):
        if self.ptr >= self.capacity:
            return False
            
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.ptr += 1
        return True
        
    def get(self):
        s = torch.FloatTensor(self.states[:self.ptr])
        a = torch.FloatTensor(self.actions[:self.ptr])
        r = torch.FloatTensor(self.rewards[:self.ptr])
        v = torch.FloatTensor(self.values[:self.ptr])
        lp = torch.FloatTensor(self.log_probs[:self.ptr])
        d = torch.FloatTensor(self.dones[:self.ptr])
        
        size = self.ptr
        self.ptr = 0
        
        return s, a, r, v, lp, d, size

class TRPOActor(nn.Module):
    """TRPO 随机 Actor Network (Gaussian Policy)"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Tanh()
            ])
            prev_dim = hidden_dim
            
        self.base = nn.Sequential(*layers)
        
        # Actor头（输出动作均值 - 无Tanh，允许高斯分布均值在任意范围）
        self.mu_head = nn.Sequential(
            nn.Linear(prev_dim, action_dim),
        )
        # 初始log_std设置较高以鼓励探索，提高下限
        self.log_std = nn.Parameter(torch.ones(action_dim) * 0.0)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
                
    def forward(self, state):
        x = self.base(state)
        mu = self.mu_head(x)
        log_std = torch.clamp(self.log_std, -1.5, 1.0)
        std = log_std.exp().expand_as(mu)
        return torch.distributions.Normal(mu, std)

class TRPOCritic(nn.Module):
    """TRPO State-Value Critic Network (V-Network)"""
    def __init__(self, state_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.Tanh()
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.v_net = nn.Sequential(*layers)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, state):
        return self.v_net(state)

# --- MATH UTILS FOR TRPO ---
def flat_grad(grads):
    """Flatten gradients into a single vector"""
    return torch.cat([g.contiguous().view(-1) for g in grads])

def get_flat_params_from(model):
    """Extract flattened parameters from a model"""
    return torch.cat([p.data.contiguous().view(-1) for p in model.parameters()])

def set_flat_params_to(model, flat_params):
    """Set flattened parameters back to a model"""
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    """
    Conjugate Gradient method to solve Ax = b.
    f_Ax: Function that computes matrix-vector product A*x
    b: Vector
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    
    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / (torch.dot(p, z) + 1e-8)
        x += v * p
        r -= v * z
        newrdotr = torch.dot(r, r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x
# ----------------------------

class TRPOAgent(BaseRLAgent, ContinuousActionMixin):
    """
    TRPO Agent for StateAwareNichePSO
    """
    ACTION_NAMES = ['w', 'c1', 'c2', 'velocity_scale']
    
    def __init__(self,
                 state_dim: int = 25,
                 action_dim: int = 4,
                 hidden_dims: List[int] = [128, 128, 64],
                 gamma: float = 0.7,
                 gae_lambda: float = 0.95,
                 max_kl: float = 0.01,
                 cg_damping: float = 0.1,
                 cg_iters: int = 10,
                 value_lr: float = 3e-4,
                 value_epochs: int = 10,
                 min_batch_size: int = 1000,
                 device: str = 'cuda'):
        super().__init__()

        self.state_dim = state_dim
        self._action_dim = action_dim
        self.hidden_dims = hidden_dims  # ← 保存 hidden_dims
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # TRPO Params
        self.max_kl = max_kl
        self.cg_damping = cg_damping
        self.cg_iters = cg_iters
        self.value_epochs = value_epochs

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Networks
        self.actor = TRPOActor(state_dim, action_dim, hidden_dims).to(self.device)
        self.critic = TRPOCritic(state_dim, hidden_dims).to(self.device)
        
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=value_lr)
        
        # Buffer (use extremely large constant instead of scalar multiply to avoid capacity issues)
        self.buffer = ExperienceBuffer(200000, state_dim, action_dim)
        
        # Action Scaling
        self.action_low = torch.tensor([b[0] for b in PSO_ACTION_BOUNDS.values()], device=self.device)
        self.action_high = torch.tensor([b[1] for b in PSO_ACTION_BOUNDS.values()], device=self.device)
        self.action_scale = (self.action_high - self.action_low) / 2
        self.action_bias = (self.action_high + self.action_low) / 2
        
    @property
    def action_type(self) -> str:
        return 'continuous'
        
    @property
    def action_dim(self) -> int:
        return self._action_dim
        
    def choose_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, Dict]:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            dist = self.actor(state_tensor)
            if deterministic or self.frozen:
                action_unsquashed = dist.mean
            else:
                action_unsquashed = dist.sample()
                
            action_tanh = torch.tanh(action_unsquashed)
            
            # Jacobian correction for log_prob
            log_prob = dist.log_prob(action_unsquashed)
            log_prob -= torch.log(self.action_scale * (1 - action_tanh.pow(2)) + 1e-6)
            log_prob = log_prob.sum(dim=-1)
            
            value = self.critic(state_tensor).squeeze(-1)
            
        action_scaled = action_tanh * self.action_scale + self.action_bias
        
        info = {
            'log_prob': log_prob.item(),
            'value': value.item(),
            'raw_action': action_unsquashed.squeeze(0).cpu().numpy()
        }
        
        return action_scaled.squeeze(0).cpu().numpy(), info
        
    def store_transition(self, state, action, reward, next_state, done, **kwargs):
        # Allow passing log_prob and value from outside for trajectory decoupling
        log_prob = kwargs.get('log_prob', 0.0)
        value = kwargs.get('value', 0.0)
        raw_action = kwargs.get('raw_action')
        
        if raw_action is None:
            # If not provided, assume action is physical and inverse map it
            action_t = torch.FloatTensor(action).to(self.device)
            act_tanh = (action_t - self.action_bias) / self.action_scale
            act_tanh = torch.clamp(act_tanh, -0.999999, 0.999999)
            raw_action = torch.atanh(act_tanh).cpu().numpy()
            
        self.buffer.add(state, raw_action, reward, value, log_prob, done)

    def _compute_gae(self, rewards, values, dones, next_value):
        advantages = torch.zeros_like(rewards).to(self.device)
        returns = torch.zeros_like(rewards).to(self.device)
        
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_val = values[t+1]
                
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            
        returns = advantages + values
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def update(self, next_value=0.0):
        if self.buffer.ptr == 0:
            return {}
            
        states, actions, rewards, values, old_log_probs, dones, size = self.buffer.get()
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        values = values.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        dones = dones.to(self.device)
        
        # 1. Compute GAE
        advantages, returns = self._compute_gae(rewards, values, dones, next_value)
        
        # NOTE: VERY IMPORTANT FOR TRPO:
        # Cache the old distribution outputs parameterized by the pre-update weights.
        # This prevents `get_kl()` from evaluating to 0 statically if `dist` is modified during `set_flat_params_to`.
        with torch.no_grad():
            old_dist_cache = self.actor(states)
            old_mu_cache = old_dist_cache.mean.detach()
            old_std_cache = old_dist_cache.stddev.detach()
        
        # 2. Compute objective function for TRPO
        def get_loss(params=None):
            if params is not None:
                set_flat_params_to(self.actor, params)
            dist = self.actor(states)
            action_tanh = torch.tanh(actions)
            log_prob = dist.log_prob(actions)
            log_prob -= torch.log(self.action_scale * (1 - action_tanh.pow(2)) + 1e-6)
            new_log_probs = log_prob.sum(dim=-1)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr_loss = (ratio * advantages).mean()
            return surr_loss
        
        # 3. Compute KL Divergence and its Hessian-Vector Product
        def get_kl():
            dist = self.actor(states)
            mu = dist.mean
            std = dist.stddev
            
            # Use strictly the cached old_std and old_mu to measure distance traveled by parameter updates
            kl = torch.log(old_std_cache / std) + (std.pow(2) + (old_mu_cache - mu).pow(2)) / (2.0 * old_std_cache.pow(2)) - 0.5
            return kl.sum(dim=-1).mean()
            
        def fisher_vector_product(v):
            kl = get_kl()
            # 1st derivative of KL w.r.t network params
            grads = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
            flat_grad_kl = flat_grad(grads)
            # KL * v
            kl_v = (flat_grad_kl * v).sum()
            # 2nd derivative of KL (Hessian vector product)
            grads = torch.autograd.grad(kl_v, self.actor.parameters())
            flat_grad_grad_kl = flat_grad(grads)
            # Add damping
            return flat_grad_grad_kl + v * self.cg_damping
            
        # 4. Compute gradient of Surrogate Objective
        surr_loss = get_loss()
        grads = torch.autograd.grad(surr_loss, self.actor.parameters())
        loss_grad = flat_grad(grads)
        
        # 5. Use Conjugate Gradient to find step direction `step_dir` = H^(-1) g
        step_dir = conjugate_gradient(fisher_vector_product, loss_grad, cg_iters=self.cg_iters)
        
        # 6. Compute step size `beta` to satisfy maximum KL constraint
        shs = 0.5 * (step_dir * fisher_vector_product(step_dir)).sum(0, keepdim=True)
        # Avoid division by zero and negative values (non-PD Hessian)
        if torch.isnan(shs) or shs <= 0:
            full_step = torch.zeros_like(step_dir)
        else:
            lm = torch.sqrt(shs / self.max_kl)
            full_step = step_dir / (lm[0] + 1e-8)
            
        # 7. Line Search
        old_actor_params = get_flat_params_from(self.actor)
        expected_improve = (loss_grad * full_step).sum(0, keepdim=True)
        
        flag = False
        fraction = 1.0
        for i in range(10): # 10 back-tracking steps
            new_params = old_actor_params + fraction * full_step
            set_flat_params_to(self.actor, new_params)
            kl = get_kl()
            new_surr_loss = get_loss()
            improve = new_surr_loss - surr_loss
            
            if kl <= self.max_kl*1.5 and improve > 0:
                flag = True
                break
            fraction *= 0.5 # Backtrack
            
        if not flag:
            set_flat_params_to(self.actor, old_actor_params)
            final_kl = 0.0
        else:
            final_kl = kl.item()
            
        # 8. Update Critic (Value Network)
        total_value_loss = 0
        for _ in range(self.value_epochs):
            v_pred = self.critic(states).squeeze(-1)
            v_loss = (v_pred - returns).pow(2).mean()
            
            self.critic_optimizer.zero_grad()
            v_loss.backward()
            self.critic_optimizer.step()
            total_value_loss += v_loss.item()
            
        return {
            'policy_loss': surr_loss.item(),
            'value_loss': total_value_loss / self.value_epochs,
            'kl_divergence': final_kl
        }
        
    def save(self, path: str):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self._action_dim,
                'hidden_dims': self.hidden_dims
            }
        }, path)
        print(f"[TRPO] Model saved to {path}")
        
    @classmethod
    def load(cls, path: str, freeze: bool = True):
        checkpoint = torch.load(path, map_location='cpu')
        config = checkpoint.get('config', {})
        agent = cls(
            state_dim=config.get('state_dim', 25),
            action_dim=config.get('action_dim', 4),
            hidden_dims=config.get('hidden_dims', [128, 128, 64])  # ← 改成旧模型的架构
        )

        agent.actor.load_state_dict(checkpoint['actor'])
        agent.critic.load_state_dict(checkpoint['critic'])
            
        if freeze:
            agent.freeze()
            agent.actor.eval()
            
        print(f"[TRPO] Loaded from {path}")
        return agent
