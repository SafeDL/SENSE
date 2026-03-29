"""
使用一个单一的全局高斯回归过程模型,并使用 Matern 核函数和 ARD 技术来提升模型的表达能力
"""
import numpy as np
import torch
import time
import gpytorch
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import scipy.io as sio
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class GPModel(gpytorch.models.ExactGP):
    """ 全局高斯过程回归模型,使用 Matern 核函数和 ARD 技术"""
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def update_GPModel(model, likelihood, X_train, Y_train):
    """
    训练/更新函数 (支持热启动) ---
    """
    train_x = torch.tensor(X_train, dtype=torch.float32).to(device)
    train_y = torch.tensor(Y_train, dtype=torch.float32).view(-1).to(device)

    if model is None:
        likelihood = GaussianLikelihood().to(device)
        likelihood.noise_covar.noise = 1e-4
        model = GPModel(train_x, train_y, likelihood).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        # Cosine Annealing 帮助跳出局部最优
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        training_iter = 200
    else:
        model.set_train_data(inputs=train_x, targets=train_y, strict=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        training_iter = 50

    model.train()
    likelihood.train()
    mll = ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        scheduler.step()
    return model, likelihood


def acquisition_function(model, likelihood, X_data, active_mask, iteration, num_sim, top_n=5):
    """
    1. Subsampling: 随机只看 N 个候选点,极大加速距离计算
    2. UCB Integration: 引入 UCB 项,强行探索高不确定性区域,打破平台期
    3. Batch Diversity: 防止采样的 top_n 个点挤在一起
    """

    # --- 候选集下采样 (Subsampling) ---
    # 不要评估所有 active_mask=False 的点,随机选 N 个点
    candidate_indices_full = np.where(~active_mask)[0]

    if len(candidate_indices_full) == 0:
        return []

    # 如果剩余点太多，随机抽样若干个点进行评估
    subsample_size = 50000
    if len(candidate_indices_full) > subsample_size:
        eval_indices_local = np.random.choice(len(candidate_indices_full), size=subsample_size, replace=False)
        candidate_indices = candidate_indices_full[eval_indices_local]
    else:
        candidate_indices = candidate_indices_full
    X_cand = torch.tensor(X_data[candidate_indices], dtype=torch.float32).to(device)
    X_train = model.train_inputs[0]

    # 使用代理模型进行预测
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = likelihood(model(X_cand))
        mean = posterior.mean
        std = posterior.stddev

    def torch_normalize(t):
        t_min, t_max = t.min(), t.max()
        if t_max - t_min < 1e-8: return torch.zeros_like(t)
        return (t - t_min) / (t_max - t_min)

    # 1. UCB Term: 即使 mean 低, 如果 std,也要去采
    # 随着迭代增加，beta 减小，从探索转向利用
    beta_ucb = 5.0 * np.exp(-4.0 * iteration / num_sim) + 0.5
    ucb_val = mean + beta_ucb * std
    term_ucb = torch_normalize(ucb_val)

    # 2. Coverage Term: 距离
    dists = torch.cdist(X_cand, X_train)
    dis_min, _ = torch.min(dists, dim=1)
    # 归一化距离
    term_cov = torch_normalize(dis_min)

    # 3. 融合得分
    # 动态权重：前期重探索(Coverage)，中期重UCB，后期重Mean
    w_ucb = 1.0
    w_cov = 0.5 * np.exp(-5.0 * iteration / num_sim)  # 快速衰减的覆盖项

    acquisition_value = w_ucb * term_ucb + w_cov * term_cov

    # 选分最高的作为待测试的样本点
    best_indices_local = []
    temp_acq = acquisition_value.clone()

    for _ in range(top_n):
        best_idx = torch.argmax(temp_acq)
        best_indices_local.append(best_idx.item())
        temp_acq[best_idx] = -float('inf')

    # 映射回全局索引
    final_indices = candidate_indices[np.array(best_indices_local)]

    return list(final_indices)


def compute_f1_final(model, likelihood, X_data, y_data):
    """
    计算全局的F1得分
    分块预测以防显存溢出 (Evaluation 阶段数据量大)
    """
    model.eval()
    likelihood.eval()

    batch_size = 5000
    n = len(X_data)
    Y_pred = np.zeros(n)

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(0, n, batch_size):
            batch_indices = slice(i, min(i + batch_size, n))
            X_batch = torch.tensor(X_data[batch_indices], dtype=torch.float32).to(device)
            preds = likelihood(model(X_batch))
            Y_pred[batch_indices] = preds.mean.cpu().numpy()

    pred_cls = (Y_pred > 0.3).astype(int)
    true_cls = (y_data.ravel() > 0.3).astype(int)
    return f1_score(true_cls, pred_cls)


def build_surrogate_iteratively(X_data, y_data):
    np.random.seed(42)
    torch.manual_seed(42)

    # 初始采样
    total_initial_samples = 500
    print(f"Initial global sampling: {total_initial_samples} samples.")

    initial_indices = np.random.choice(len(X_data), size=total_initial_samples, replace=False)
    active_mask = np.zeros(len(X_data), dtype=bool)
    active_mask[initial_indices] = True

    # 初始训练集
    X_train = X_data[initial_indices]
    Y_train = y_data[initial_indices]

    # 模型容器
    model = None
    likelihood = None

    num_sim = 500
    f1_scores = []

    print(f"Start Single Global GP Active Learning on {device}...")
    pbar = tqdm(range(num_sim))

    for iteration in pbar:
        # A. 训练 (全量更新)
        model, likelihood = update_GPModel(model, likelihood, X_train, Y_train)

        # B. 评估
        f1 = compute_f1_final(model, likelihood, X_data, y_data)
        f1_scores.append(f1)
        pbar.set_description(f"Iter {iteration}: F1={f1:.4f}, Samples={len(X_train)}")

        # C. 采样 (全局)
        new_indices = acquisition_function(model, likelihood, X_data, active_mask,iteration, num_sim, top_n=5)

        if not new_indices: break

        # D. 更新数据
        for idx in new_indices:
            active_mask[idx] = True

        # 重新构建训练集 (简单堆叠)
        X_train = np.vstack([X_train, X_data[new_indices]])
        Y_train = np.vstack([Y_train, y_data[new_indices]])

    # 保存训练好的代理模型
    save_model(model, likelihood, '../../results/s5exp/surrogate_model_compare.pkl')
    
    # 返回训练历史用于对比 (使用迭代次数作为横坐标)
    iterations = list(range(0, len(f1_scores)))
    return iterations, f1_scores


def save_model(model, likelihood, filename):
    model.eval()
    likelihood.eval()
    # 获取训练数据 (从模型内部获取，确保与模型状态一致)
    X_train = model.train_inputs[0].cpu().numpy()
    Y_train = model.train_targets.cpu().numpy().reshape(-1, 1)
    serializable_model = {
        'model_state': model.cpu().state_dict(),
        'likelihood_state': likelihood.cpu().state_dict(),
        'X_train': X_train,
        'Y_train': Y_train,
        'kernel_type': 'MaternKernel(nu=2.5)',
        'ard_num_dims': X_train.shape[-1]
    }
    with open(filename, 'wb') as f:
        pickle.dump(serializable_model, f)
        print(f"Global surrogate GP model has been saved to {filename}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open('./train_data/scenario05_grid_x.pkl', 'rb') as f:
        grid_x = pickle.load(f)
        grid_x = grid_x[:80000,:]
    with open('./train_data/scenario05_grid_y.pkl', 'rb') as f:
        grid_y = np.array(pickle.load(f)).reshape(-1, 1)
    print(f"Data Loaded: X={grid_x.shape}, Y={grid_y.shape}")
    start_time = time.time()
    iterations, f1_scores = build_surrogate_iteratively(X_data=grid_x, y_data=grid_y)
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    
    # 保存GP模型的训练历史到.mat文件
    gp_results = {
        'iterations': np.array(iterations),
        'gp_f1': np.array(f1_scores)
    }
    sio.savemat('../../../matlab_scripts/gp_surrogate_result.mat', gp_results)
    print(f"GP training history saved to matlab_scripts/gp_surrogate_result.mat")
