"""
加载训练好的GPU版代理模型,并验证其性能
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import gpytorch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from rlsan.src.surrogate.utils import load_surrogate_model


def global_predict(model, likelihood, X_data, batch_size=5000):
    """对全量数据进行分批预测"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = len(X_data)
    mean_preds = np.zeros(n)
    var_preds = np.zeros(n)

    model.eval()
    likelihood.eval()

    print(f"Starting global prediction on {n} samples...")
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(0, n, batch_size):
            batch_indices = slice(i, min(i + batch_size, n))
            X_batch = torch.tensor(X_data[batch_indices], dtype=torch.float32).to(device)
            # 预测
            posterior = likelihood(model(X_batch))
            mean_preds[batch_indices] = posterior.mean.cpu().numpy()
            var_preds[batch_indices] = posterior.variance.cpu().numpy()
    return mean_preds, var_preds


def compute_f1(Y_pred, Y_test, threshold=0.3):
    pred_cls = (Y_pred > threshold).astype(int)
    true_cls = (Y_test > threshold).astype(int)
    print(f"length of true collisions: {sum(true_cls)}")
    print(f"length of predicted collisions: {sum(pred_cls)}")

    # 捕获的碰撞案例数 (True Positive)
    collision_captured = np.sum(pred_cls * true_cls)
    print(f"Number of collisions captured: {collision_captured}")

    f1 = f1_score(true_cls, pred_cls)
    precision = np.sum(pred_cls * true_cls) / (np.sum(pred_cls) + 1e-8)
    recall = np.sum(pred_cls * true_cls) / (np.sum(true_cls) + 1e-8)
    return precision, recall, f1


if __name__=='__main__':
    # 1. 加载模型
    model_path = '../../results/s1exp/surrogate_model_1000.pkl'
    model, likelihood = load_surrogate_model(model_path)
    ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().flatten()
    print(f"Lengthscale: {ls}")

    # 2. 加载数据
    with open('../../results/grid/scenario01/grid_x.pkl', 'rb') as f:
        X_data = pickle.load(f)
    with open('./train_data/scenario01_grid_y.pkl', 'rb') as f:
        y_data = np.array(pickle.load(f)).reshape(-1, 1)

    print(f"Data Loaded: X={X_data.shape}, Y={y_data.shape}")

    # 3. 绘制真实得分分布
    plt.figure(figsize=(10, 6))
    plt.hist(y_data, bins=50, alpha=0.7, color='green', edgecolor='black', label='Ground Truth')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 4. 执行预测 (替代原本的 joblib parallel)
    Y_pred, Variance = global_predict(model, likelihood, X_data,)

    # 5. 计算指标(假设 Score > 0.3 为危险，需与训练时一致)
    precision, recall, f1 = compute_f1(Y_pred, y_data.ravel(), threshold=0.3)

    print(f"\nEvaluation Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # 6. 打印训练样本占比
    train_size = len(model.train_inputs[0])
    print(f"\nSurrogate Model Details:")
    print(f"Training Samples Used: {train_size}")
    print(f"Total Test Space Size: {len(X_data)}")
    print(f"Training Proportion:   {train_size / len(X_data):.2%}")
