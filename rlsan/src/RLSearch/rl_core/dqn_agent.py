"""
dqn_agent.py - 通用化 Double DQN Agent
支持预训练权重的保存和加载，用于跨域迁移
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Optional, Tuple, Dict, Any

# 支持包导入和直接运行两种模式
try:
    from .networks import EnhancedDQN
except (ImportError, ValueError):
    from networks import EnhancedDQN

# 自动检测设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DoubleDQNAgent:
    """
    通用化 Double DQN Agent
    
    支持:
    - 预训练权重的保存/加载
    - 冻结模式（部署时不更新）
    - 可配置的网络结构
    
    Args:
        state_dim: 状态维度
        num_actions: 动作数量
        alpha: 学习率
        gamma: 折扣因子
        epsilon: 初始探索率
        epsilon_decay: 探索率衰减
        min_epsilon: 最小探索率
        buffer_size: 经验回放缓冲区大小
        batch_size: 批次大小
        hidden_dims: 隐藏层维度列表
    """
    def __init__(self, 
                 state_dim: int, 
                 num_actions: int, 
                 alpha: float = 0.0003, 
                 gamma: float = 0.6,
                 epsilon: float = 1.0, 
                 epsilon_decay: float = 0.995, 
                 min_epsilon: float = 0.1,
                 temperature: float = 2.0,  # 初始温度
                 buffer_size: int = 5000,
                 batch_size: int = 128,
                 hidden_dims: Optional[list] = None):
        
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.temperature = temperature
        self.batch_size = batch_size
        self.hidden_dims = hidden_dims or [256, 256, 128]
        
        # 是否冻结（部署模式）
        self.frozen = False
        
        # 创建网络
        self.model = EnhancedDQN(state_dim, num_actions, self.hidden_dims).to(device)
        self.target_model = EnhancedDQN(state_dim, num_actions, self.hidden_dims).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        # 优化器和学习率调度器
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        
        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # 训练统计
        self.train_steps = 0
        self.total_loss = 0.0
    
    def choose_action(self, state: np.ndarray, deterministic: bool = False, temperature: float = 1.0) -> Tuple[int, Optional[np.ndarray]]:
        """
        根据当前策略选择动作
        
        Args:
            state: 当前状态
            deterministic: 是否确定性选择（用于评估）
            temperature: Softmax 温度系数
            
        Returns:
            action: 选择的动作索引
            probs: 动作概率分布 (Softmax output)
        """
        # 部署模式或确定性模式下不探索
        if self.frozen or deterministic:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = self.model(state_tensor).squeeze(0)
                action = q_values.argmax().item()
                
                # 构造 One-Hot 概率 (确定性)
                probs = np.zeros(self.num_actions, dtype=np.float32)
                probs[action] = 1.0
                return action, probs
                
        # Epsilon-Greedy 保证最低限度的均匀探索
        if random.random() < self.epsilon:
            action = random.randrange(self.num_actions)
            # 探索时构造近似均匀概率以供可视化
            probs = np.ones(self.num_actions, dtype=np.float32) / self.num_actions
            return action, probs
        
        # Softmax 探索 (Boltzmann) 使用自身保存的温度
        actual_temperature = self.temperature if temperature == 1.0 else temperature
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = self.model(state_tensor).squeeze(0)
            
            # 温度缩放
            scaled_q = q_values / (actual_temperature + 1e-8)
            # 数值稳定 Softmax
            probs = torch.softmax(scaled_q, dim=0).cpu().numpy()
            
            # 按概率采样
            try:
                action = np.random.choice(len(probs), p=probs)
            except ValueError:
                action = probs.argmax()  # Fallback if probs sum to nan
            
            return action, probs
    
    def store_transition(self, state: np.ndarray, action: int, 
                        reward: float, next_state: np.ndarray, done: bool = False):
        """
        存储一个转移到经验回放缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
        """
        if not self.frozen:
            self.replay_buffer.append((state, action, reward, next_state, done))
    
    def update_q_values(self) -> Tuple[Optional[float], Optional[float]]:
        """
        从经验回放缓冲区采样并更新 Q 值
        
        Returns:
            loss: 训练损失（如果更新了）
            avg_q: 平均 Q 值（如果更新了）
        """
        if self.frozen:
            return None, None
        
        if len(self.replay_buffer) < self.batch_size:
            return None, None
        
        # 采样批次
        batch = random.sample(self.replay_buffer, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        
        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32).to(device)
        action_batch = torch.tensor(action_batch, dtype=torch.long).to(device)
        reward_batch = torch.tensor(np.array(reward_batch), dtype=torch.float32).to(device)
        next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float32).to(device)
        done_batch = torch.tensor(np.array(done_batch), dtype=torch.float32).to(device)
        
        # 计算当前 Q 值
        q_values = self.model(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        # Double DQN: 用主网络选动作，用目标网络估值
        with torch.no_grad():
            next_actions = self.model(next_state_batch).argmax(1)
            next_q_values = self.target_model(next_state_batch).gather(
                1, next_actions.unsqueeze(1)).squeeze(1)
            
            # ===== 关键修复：裁剪 target Q-value 防止过估计 =====
            max_q_value = 50.0  # Q值上限
            next_q_values = torch.clamp(next_q_values, -5.0, max_q_value)
            
            # Bellman backup: correctly account for terminal transition (done = 1.0)
            target_q_values = reward_batch + (1.0 - done_batch) * self.gamma * next_q_values
            # 同样裁剪 target
            target_q_values = torch.clamp(target_q_values, -5.0, max_q_value * 2)
        
        # 计算损失并更新
        loss = nn.SmoothL1Loss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.train_steps += 1
        self.total_loss += loss.item()
        
        return loss.item(), q_values.mean().item()
    
    def update_target_network(self):
        """硬更新目标网络"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def soft_update_target_network(self, tau: float = 0.005):
        """软更新目标网络"""
        for target_param, param in zip(self.target_model.parameters(), 
                                       self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def decay_epsilon(self):
        """衰减探索率"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.scheduler.step()
    
    def freeze(self):
        """
        冻结 Agent（部署模式）
        停止更新网络和经验回放
        """
        self.frozen = True
        self.model.eval()
        self.epsilon = 0.0  # 不再探索
        print("[DoubleDQNAgent] Frozen for deployment (no updates, no exploration)")
    
    def unfreeze(self):
        """解冻 Agent（继续训练）"""
        self.frozen = False
        self.model.train()
        print("[DoubleDQNAgent] Unfrozen for training")
    
    def save(self, path: str):
        """
        保存 Agent 权重和配置
        
        Args:
            path: 保存路径（.pth 文件）
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': {
                'state_dim': self.state_dim,
                'num_actions': self.num_actions,
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay,
                'min_epsilon': self.min_epsilon,
                'temperature': self.temperature,
                'hidden_dims': self.hidden_dims,
            },
            'train_steps': self.train_steps,
        }
        torch.save(checkpoint, path)
        print(f"[DoubleDQNAgent] Saved to {path}")
    
    @classmethod
    def load(cls, path: str, freeze: bool = True) -> 'DoubleDQNAgent':
        """
        加载 Agent 权重
        
        Args:
            path: 权重文件路径
            freeze: 是否冻结（用于部署）
            
        Returns:
            agent: 加载的 Agent 实例
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        agent = cls(
            state_dim=config['state_dim'],
            num_actions=config['num_actions'],
            alpha=config['alpha'],
            gamma=config['gamma'],
            epsilon=config.get('epsilon', 0.1),
            epsilon_decay=config.get('epsilon_decay', 0.995),
            min_epsilon=config.get('min_epsilon', 0.1),
            temperature=config.get('temperature', 2.0),
            hidden_dims=config.get('hidden_dims', [256, 256, 128]),
        )
        
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.target_model.load_state_dict(checkpoint['target_model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        agent.train_steps = checkpoint.get('train_steps', 0)
        
        if freeze:
            agent.freeze()
        
        print(f"[DoubleDQNAgent] Loaded from {path} (freeze={freeze})")
        return agent
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        获取给定状态的所有 Q 值
        
        Args:
            state: 当前状态
            
        Returns:
            q_values: 所有动作的 Q 值
        """
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = self.model(state_tensor).squeeze(0).cpu().numpy()
        return q_values
    
    def choose_action_batch(self, states: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        批量选择动作 (Softmax/Boltzmann)
        
        Args:
            states: shape (batch_size, state_dim) 的状态批量
            deterministic: 是否使用确定性策略
            
        Returns:
            actions: shape (batch_size,) 的动作索引
        """
        # 部署模式或确定性模式下不探索
        if self.frozen or deterministic:
            with torch.no_grad():
                state_tensor = torch.tensor(states, dtype=torch.float32).to(device)
                q_values = self.model(state_tensor)
                return q_values.argmax(dim=1).cpu().numpy()
        
        # Softmax 探索 (Boltzmann)
        with torch.no_grad():
            state_tensor = torch.tensor(states, dtype=torch.float32).to(device)
            q_values = self.model(state_tensor)
            
            # 温度缩放
            scaled_q = q_values / (self.temperature + 1e-8)
            # Softmax
            probs = torch.softmax(scaled_q, dim=1)
            
            # 批量多项式采样
            actions = torch.multinomial(probs, num_samples=1).squeeze(1).cpu().numpy()
            
        return actions
    
    def store_transition_batch(self, states: np.ndarray, actions: np.ndarray,
                               rewards: np.ndarray, next_states: np.ndarray,
                               dones: Optional[np.ndarray] = None):
        """
        批量存储转移
        
        Args:
            states: shape (batch_size, state_dim)
            actions: shape (batch_size,)
            rewards: shape (batch_size,)
            next_states: shape (batch_size, state_dim)
            dones: shape (batch_size,) optional
        """
        if not self.frozen:
            for i in range(len(states)):
                done = dones[i] if dones is not None else False
                self.replay_buffer.append((states[i], actions[i], rewards[i], next_states[i], done))
    
    def get_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        return {
            'train_steps': self.train_steps,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': self.total_loss / max(1, self.train_steps),
            'frozen': self.frozen,
        }
