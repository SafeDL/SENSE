"""
networks.py - 通用神经网络定义
包含 MLP 和增强型 DQN 网络结构
"""

import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    """
    通用多层感知机，带 LayerNorm 和可选残差连接
    
    Args:
        input_dim: 输入维度
        hidden_dims: 隐藏层维度列表
        output_dim: 输出维度
        use_layer_norm: 是否使用 LayerNorm
        use_residual: 是否使用残差连接（需要相邻隐藏层维度相同）
        activation: 激活函数类型
    """
    def __init__(self, input_dim, hidden_dims, output_dim, 
                 use_layer_norm=True, use_residual=True, activation='relu'):
        super(MLP, self).__init__()
        
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        else:
            self.activation = nn.ReLU()
        
        # 构建层
        layers = []
        layer_norms = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layer_norms.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.ModuleList(layers)
        self.layer_norms = nn.ModuleList(layer_norms) if use_layer_norm else None
        self.output_layer = nn.Linear(prev_dim, output_dim)
        self.hidden_dims = hidden_dims
    
    def forward(self, x):
        prev_x = None
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            x = self.activation(x)
            
            # 残差连接（仅当维度匹配时）
            if self.use_residual and prev_x is not None and prev_x.shape == x.shape:
                x = x + prev_x
            prev_x = x
        
        return self.output_layer(x)


class EnhancedDQN(nn.Module):
    """
    增强型 DQN 网络
    带有 LayerNorm 和残差连接，适用于 Double DQN
    
    Args:
        state_dim: 状态维度
        num_actions: 动作数量
        hidden_dims: 隐藏层维度，默认 [256, 256, 128]
    """
    def __init__(self, state_dim, num_actions, hidden_dims=None):
        super(EnhancedDQN, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256, 128]
        
        self.fc1 = nn.Linear(state_dim, hidden_dims[0])
        self.ln1 = nn.LayerNorm(hidden_dims[0])
        
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.ln2 = nn.LayerNorm(hidden_dims[1])
        
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.ln3 = nn.LayerNorm(hidden_dims[2])
        
        self.fc4 = nn.Linear(hidden_dims[2], num_actions)
        
        # 存储维度信息
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden_dims = hidden_dims

    def forward(self, x):
        x1 = torch.relu(self.ln1(self.fc1(x)))
        x2 = torch.relu(self.ln2(self.fc2(x1)))
        x2 = x2 + x1  # 残差连接（维度相同时）
        x3 = torch.relu(self.ln3(self.fc3(x2)))
        return self.fc4(x3)


class SinusoidalStateEncoder:
    """
    多尺度正弦变换器
    将微小的状态变化放大为显著的神经网络输入
    
    参考: Yin et al. (2023) - Positional Encoding for RL
    
    对每个状态分量 x 进行: [sin(x * 2^0 * π), sin(x * 2^1 * π), ...]
    
    Args:
        num_scales: 正弦变换的尺度数量
    """
    def __init__(self, num_scales=4):
        self.num_scales = num_scales
        self.scales = np.array([2 ** i for i in range(num_scales)])
    
    def encode(self, state):
        """
        编码状态向量
        
        Args:
            state: shape (D,) 的状态向量
            
        Returns:
            encoded: shape (D * num_scales,) 的编码向量
        """
        encoded = []
        for s in self.scales:
            encoded.append(np.sin(state * s * np.pi))
        return np.concatenate(encoded)
    
    def encode_batch(self, states):
        """
        批量编码状态
        
        Args:
            states: shape (N, D) 的状态矩阵
            
        Returns:
            encoded: shape (N, D * num_scales) 的编码矩阵
        """
        N, D = states.shape
        encoded = np.zeros((N, D * self.num_scales))
        for i, s in enumerate(self.scales):
            encoded[:, i*D:(i+1)*D] = np.sin(states * s * np.pi)
        return encoded
    
    def get_output_dim(self, input_dim):
        """返回编码后的维度"""
        return input_dim * self.num_scales


class RunningMeanStd:
    """
    运行时均值和标准差估计器
    用于状态归一化
    
    Args:
        shape: 输入状态的形状
        epsilon: 防止除零的小常数
    """
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.epsilon = epsilon
    
    def update(self, x):
        """
        使用新数据更新统计量
        
        Args:
            x: shape (batch_size, *shape) 或 (*shape,) 的数据
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        new_var = M2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
    
    def normalize(self, x):
        """归一化输入数据"""
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)
    
    def denormalize(self, x):
        """反归一化"""
        return x * np.sqrt(self.var + self.epsilon) + self.mean
