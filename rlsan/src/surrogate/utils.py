import gpytorch
import torch
import pickle


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


def load_surrogate_model(filename):
    # 自动选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {filename} on {device}...")

    # 读取 pickle
    with open(filename, 'rb') as f:
        saved_data = pickle.load(f)

    # 1. 从保存的数据中恢复 X_train 和 Y_train
    # 这是 ExactGP 初始化的必要步骤
    X_numpy = saved_data['X_train']
    Y_numpy = saved_data['Y_train']

    train_x = torch.tensor(X_numpy, dtype=torch.float32).to(device)
    train_y = torch.tensor(Y_numpy, dtype=torch.float32).view(-1).to(device)

    # 2. 初始化模型结构
    # 使用上面显式定义的 GPModel，它会自动根据 train_x.shape[-1] 设置 ARD 参数
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = GPModel(train_x, train_y, likelihood).to(device)

    # 3. 加载权重
    # 因为类定义现在匹配了，load_state_dict 不会再报 Missing key 或 Unexpected key
    try:
        model.load_state_dict(saved_data['model_state'])
        likelihood.load_state_dict(saved_data['likelihood_state'])
    except RuntimeError as e:
        print("错误：权重加载失败。请确认上方的 GPModel 类定义与训练代码完全一致。")
        raise e

    # 4. 设置为评估模式
    model.eval()
    likelihood.eval()

    print(f"Model loaded successfully. Training samples: {len(train_x)}")
    return model, likelihood


def save_surrogate_model(model, likelihood, filename):
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
