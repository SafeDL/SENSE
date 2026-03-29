import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import random
import pickle

# ==========================================
# 绘图标准配置
# ==========================================
def set_ieee_style():
    # 字号配置 (IEEE 标准：正文约 10pt，图中文字通常 8-10pt)
    plt.rcParams['axes.labelsize'] = 10  # 轴标签
    plt.rcParams['font.size'] = 10  # 图例、标题等
    plt.rcParams['legend.fontsize'] = 9  # 图例
    plt.rcParams['xtick.labelsize'] = 9  # X轴刻度
    plt.rcParams['ytick.labelsize'] = 9  # Y轴刻度

    # 3. 线条与点配置
    plt.rcParams['lines.linewidth'] = 1.0  # 线宽
    plt.rcParams['axes.linewidth'] = 0.8  # 坐标轴线宽
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['grid.linewidth'] = 0.5
    plt.rcParams['grid.alpha'] = 0.5

    # 4. 导出配置
    plt.rcParams['savefig.dpi'] = 600  # 高分辨率 (投稿要求通常 >300)
    plt.rcParams['savefig.bbox'] = 'tight'  # 自动切除白边
    plt.rcParams['figure.autolayout'] = True

def set_seed(seed: int):
    """设置随机种子"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def load_surrogate_model(model_path: str):
    """加载代理模型"""
    try:
        from rlsan.src.surrogate.utils import load_surrogate_model as load_gp
        return load_gp(model_path)
    except ImportError:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                return data['model'], data['likelihood']
            elif isinstance(data, tuple):
                return data[0], data[1]
            else:
                raise ValueError("Unknown model format")

def get_dummy_surrogate_model(dim: int):
    """获取用于测试的随机GP代理模型"""
    import gpytorch
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            try:
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(ard_num_dims=dim)
                )
            except TypeError:
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x))

    train_x, train_y = torch.randn(10, dim), torch.randn(10)
    gp_likelihood = gpytorch.likelihoods.GaussianLikelihood()
    gp_model = ExactGPModel(train_x, train_y, gp_likelihood)
    return gp_model, gp_likelihood

# ==========================================
# RL 评测统一指标
# ==========================================
def compute_rl_metrics(all_rewards, all_action_means, pso, window=20):
    """
    计算用于比较不同 RL 算法在驱动 Niche PSO 时的4个核心评估指标。
    
    Args:
        all_rewards: 整个训练周期的 mean reward 列表
        all_action_means: 每回合平均动作参数的列表
        pso: NichePSO 优化器实例 (用于提取发现的危险小生境数)
        window: 用于平滑的滑动窗口大小 (默认20回合)
    """
    if len(all_rewards) == 0:
        return {}
    
    # 1. Convergence Efficiency (训练收敛性: 最后 window 个 episode 的平均 reward)
    convergence_efficiency = np.mean(all_rewards[-window:]) if len(all_rewards) >= window else np.mean(all_rewards)
    
    # 2. Niche Diversity Potential (搜索多样性潜力: 触发小生境且成功发现的失败域数量)
    niche_diversity = 0
    if hasattr(pso, 'reward_calculator') and pso.reward_calculator:
        if hasattr(pso.reward_calculator, 'danger_tracker'):
            niche_diversity = pso.reward_calculator.danger_tracker.get_num_niches()
            
    # 3. Action Entropy & Stability (动作决策稳定性: 最近 window 个 episode 动作的平均标准差)
    # 对于连续动作，这代表了搜索是否收敛到了一个稳定的控制策略。值越低说明越稳定。
    if all_action_means and len(all_action_means) >= min(window, 5):
        # 确保动作列表中的所有元素维度一致
        try:
            recent_actions = np.array(all_action_means[-window:])
            if len(recent_actions.shape) == 2:
                action_stability = np.mean(np.std(recent_actions, axis=0))
            else:
                action_stability = 0.0
        except Exception:
            action_stability = 0.0
    else:
        action_stability = 0.0
        
    # 4. Steps to Convergence (收敛迭代次数: 平均奖励变动率 < 5% 时的迭代次数)
    steps_to_convergence = len(all_rewards)
    if len(all_rewards) >= window:
        ma_rewards = np.convolve(all_rewards, np.ones(window)/window, mode='valid')
        if len(ma_rewards) > 0:
            final_reward = ma_rewards[-1]
            # 为了避免除以 0，限制最小分母为 1.0 或依据 reward 尺度
            denom = abs(final_reward) if abs(final_reward) > 1e-3 else 1.0
            
            for i in range(len(ma_rewards)):
                # 如果从该点往后的所有滑动平均变动率都不超过 5%
                if np.all(np.abs(ma_rewards[i:] - final_reward) / denom < 0.05):
                    # i 是 mode='valid' 下的索引，对应原数组真实回合是 i + window - 1
                    steps_to_convergence = i + window - 1
                    break

    print("\n" + "="*50)
    print("🎯 RL Strategy Evaluation Metrics 🎯")
    print("="*50)
    print(f"1. Convergence Efficiency : {convergence_efficiency:.4f} (Avg reward of last {min(window, len(all_rewards))} eps)")
    print(f"2. Niche Diversity Potential: {niche_diversity} (Total unique failure domains found)")
    print(f"3. Action Decision Stability: {action_stability:.4f} (Lower std_dev means policy stabilized)")
    print(f"4. Steps to Convergence   : {steps_to_convergence} (Episodes to reach within 5% of final MA)")
    print("="*50 + "\n")
    
    return {
        'convergence_efficiency': convergence_efficiency,
        'niche_diversity_potential': niche_diversity,
        'action_stability': action_stability,
        'steps_to_convergence': steps_to_convergence
    }
