"""
ad_scenario_env.py - 自动驾驶场景环境
封装 GP 代理模型和 CARLA 仿真接口，用于在线迁移阶段
"""

import torch
import numpy as np
from typing import Tuple, Optional, Any
from .base_env import BaseOptEnv
import gpytorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ADScenarioEnv(BaseOptEnv):
    """
    自动驾驶场景环境
    
    用于 Stage 2 在线迁移，封装:
    - GP 代理模型（快速预测）
    - CARLA 仿真器（高保真验证）
    
    Args:
        gp_model: GPyTorch 高斯过程模型
        gp_likelihood: GPyTorch 似然函数
        runner: CARLA 仿真运行器（可选）
        dim: 搜索空间维度（测试参数数量）
        bounds: 搜索空间边界
        uncertainty_threshold: 触发真实仿真的不确定性阈值
        max_real_calls_per_iter: 每次迭代最多调用真实仿真的次数
    """
    
    def __init__(self, 
                 gp_model: Any,
                 gp_likelihood: Any,
                 runner: Optional[Any] = None,
                 dim: int = 3,
                 bounds: Tuple[float, float] = (-1, 1),
                 uncertainty_threshold: float = 0.05,
                 max_real_calls_per_iter: int = 5):
        
        super().__init__(dim, bounds)
        
        self.gp_model = gp_model
        self.gp_likelihood = gp_likelihood
        self.runner = runner
        
        self.uncertainty_threshold = uncertainty_threshold
        self.max_real_calls_per_iter = max_real_calls_per_iter
        
        # 统计信息
        self.real_simulation_count = 0
        self.surrogate_call_count = 0
        
        # 将模型移到正确的设备
        if self.gp_model is not None:
            self._setup_gp_model()
    
    def _setup_gp_model(self):
        """设置 GP 模型到正确的设备"""
        try:
            self.gp_model.to(device)
            self.gp_likelihood.to(device)
            
            if hasattr(self.gp_model, 'train_inputs') and self.gp_model.train_inputs is not None:
                self.gp_model.train_inputs = tuple(t.to(device) for t in self.gp_model.train_inputs)
            if hasattr(self.gp_model, 'train_targets') and self.gp_model.train_targets is not None:
                self.gp_model.train_targets = self.gp_model.train_targets.to(device)
        except Exception as e:
            print(f"Warning: Failed to move GP model to {device}: {e}")
    
    def evaluate(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        评估粒子位置
        
        优先使用 GP 代理模型预测，对于高不确定性点可选择调用真实仿真
        
        Args:
            positions: shape (N, dim) 的粒子位置
            
        Returns:
            fitness: shape (N,) 的适应度值（危险度的负值，即越小越危险）
            uncertainty: shape (N,) 的不确定性值
        """
        positions = np.atleast_2d(positions)
        self.evaluation_count += len(positions)
        self.surrogate_call_count += len(positions)
        
        # 使用 GP 模型预测
        fitness, uncertainty = self._predict_with_gp(positions)
        
        # 可选：对高不确定性点使用真实仿真
        if self.runner is not None:
            high_uncertainty_mask = uncertainty > self.uncertainty_threshold
            if np.any(high_uncertainty_mask):
                fitness, uncertainty = self._refine_with_simulation(
                    positions, fitness, uncertainty, high_uncertainty_mask
                )
        
        return fitness, uncertainty
    
    def _predict_with_gp(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用 GP 模型预测
        
        Args:
            positions: 粒子位置
            
        Returns:
            fitness: 适应度（负的预测均值）
            uncertainty: 预测方差
        """
        x_tensor = torch.tensor(positions, dtype=torch.float32).to(device)
        
        self.gp_model.eval()
        self.gp_likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.gp_likelihood(self.gp_model(x_tensor))
            mu = posterior.mean.cpu().numpy()
            sigma2 = posterior.variance.cpu().numpy()
        
        # 适应度 = 负的预测均值（最小化问题，值越小越"危险"）
        # 注: 原始危险度越高越危险，这里取负值转换为最小化
        fitness = -mu - 1.0 * sigma2  # LCB 风格：探索不确定区域
        
        return fitness, sigma2
    
    def _refine_with_simulation(self, positions: np.ndarray, 
                                 fitness: np.ndarray, 
                                 uncertainty: np.ndarray,
                                 high_uncertainty_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        对高不确定性点使用真实仿真精炼
        
        Args:
            positions: 粒子位置
            fitness: GP 预测的适应度
            uncertainty: GP 预测的不确定性
            high_uncertainty_mask: 高不确定性点的掩码
            
        Returns:
            更新后的 fitness 和 uncertainty
        """
        selected_idx = np.where(high_uncertainty_mask)[0]
        
        # 限制每次调用的数量
        if len(selected_idx) > self.max_real_calls_per_iter:
            # 优先选择不确定性最高的点
            uncertainties = uncertainty[selected_idx]
            top_idx = np.argsort(uncertainties)[-self.max_real_calls_per_iter:]
            selected_idx = selected_idx[top_idx]
        
        if len(selected_idx) == 0:
            return fitness, uncertainty
        
        print(f"[ADScenarioEnv] High uncertainty in {len(selected_idx)} samples. Calling real simulation...")
        
        # 调用真实仿真
        selected_positions = positions[selected_idx]
        true_values = self._call_real_simulation(selected_positions)
        
        # 更新结果
        fitness[selected_idx] = -true_values  # 转换为最小化问题
        uncertainty[selected_idx] = 0  # 真实值无不确定性
        
        # 可选：更新代理模型
        self._update_surrogate_model(selected_positions, true_values)
        
        return fitness, uncertainty
    
    def _call_real_simulation(self, positions: np.ndarray) -> np.ndarray:
        """
        调用真实 CARLA 仿真
        
        Args:
            positions: 测试参数位置
            
        Returns:
            模拟结果（危险度）
        """
        results = []
        
        for idx in range(len(positions)):
            self.real_simulation_count += 1
            try:
                result, _ = self.runner.run(positions[idx].reshape(1, -1))
                results.append(result)
            except Exception as e:
                print(f"Warning: Simulation failed for position {idx}: {e}")
                results.append(0.0)  # 默认值
        
        return np.array(results).flatten()
    
    def _update_surrogate_model(self, new_positions: np.ndarray, new_values: np.ndarray):
        """
        使用新数据更新代理模型
        
        Args:
            new_positions: 新的测试参数
            new_values: 对应的真实仿真结果
        """
        try:
            new_X = torch.tensor(new_positions, dtype=torch.float32).to(device)
            new_Y = torch.tensor(new_values, dtype=torch.float32).reshape(-1).to(device)
            
            old_X = self.gp_model.train_inputs[0]
            old_Y = self.gp_model.train_targets
            
            train_x = torch.cat([old_X, new_X])
            train_y = torch.cat([old_Y, new_Y])
            
            self.gp_model.set_train_data(inputs=train_x, targets=train_y, strict=False)
            
            print(f"[ADScenarioEnv] Surrogate model updated with {len(new_positions)} new samples")
        except Exception as e:
            print(f"Warning: Failed to update surrogate model: {e}")
    
    def evaluate_dangerous_level(self, positions: np.ndarray) -> np.ndarray:
        """
        直接获取危险度（不取负值）
        
        Args:
            positions: 测试参数位置
            
        Returns:
            危险度值（越大越危险）
        """
        positions = np.atleast_2d(positions)
        x_tensor = torch.tensor(positions, dtype=torch.float32).to(device)

        self.gp_model.eval()
        self.gp_likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.gp_likelihood(self.gp_model(x_tensor))
            mu = posterior.mean.cpu().numpy()
        
        return mu
    
    def get_uncertainty(self, positions: np.ndarray) -> np.ndarray:
        """
        获取预测不确定性
        
        Args:
            positions: 测试参数位置
            
        Returns:
            预测方差
        """
        positions = np.atleast_2d(positions)
        x_tensor = torch.tensor(positions, dtype=torch.float32).to(device)
        self.gp_model.eval()
        self.gp_likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.gp_likelihood(self.gp_model(x_tensor))
            sigma2 = posterior.variance.cpu().numpy()
        
        return sigma2
    
    def get_info(self) -> dict:
        """返回环境信息"""
        info = super().get_info()
        info.update({
            'real_simulation_count': self.real_simulation_count,
            'surrogate_call_count': self.surrogate_call_count,
            'uncertainty_threshold': self.uncertainty_threshold,
            'has_runner': self.runner is not None,
        })
        return info
    
    def reset_counters(self):
        """重置统计计数器"""
        self.real_simulation_count = 0
        self.surrogate_call_count = 0
        self.evaluation_count = 0
    
    # ========== 训练脚本兼容性接口 ==========
    
    def get_current_function_name(self) -> str:
        """返回当前环境名称"""
        return "ad_scenario"
    
    def set_random_function(self, *args, **kwargs):
        """兼容接口（无操作，ADS 场景不需要切换函数）"""
        pass
    
    def set_function(self, name: str):
        """兼容接口（无操作）"""
        pass
    
    def reset_search_space(self):
        """
        重置搜索起点（用于训练时周期性重采样）
        
        这个方法可以用于在训练过程中模拟"切换函数"的效果，
        通过重置评估计数来开始新的搜索轮次
        """
        self.evaluation_count = 0
        self.surrogate_call_count = 0
