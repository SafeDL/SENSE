from typing import final
import pandas as pd
import random
from pyDOE import lhs
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
import scipy.io as sio
import time
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
import warnings
import itertools

warnings.simplefilter(action='ignore', category=UserWarning)


def latin_sampling(parameters, num_samples=100):
    """
    根据给定的参数范围进行拉丁超立方采样
    """
    min_values = []  # 最小值
    max_values = []  # 最大值
    for key, value in parameters.items():
        min_values.append(value[0])
        max_values.append(value[1])

    # 进行拉丁超立方采样,在已知危险用例周围的空间维度生成10000个样本
    samples = lhs(n=len(min_values), samples=num_samples, criterion='c')
    # 将采样样本映射到给定的数组范围
    scaled_samples = np.array(min_values) + (np.array(max_values) - np.array(min_values)) * samples
    # 打印采样结果
    # print(scaled_samples)

    return scaled_samples.astype(np.float32)


def Grid_Search(parameters, step=10):
    # 定义每个变量的取值范围
    test_parameters = {'x1': np.linspace(parameters['x1'][0], parameters['x1'][1], step).tolist(),
                       'x2': np.linspace(parameters['x2'][0], parameters['x2'][1], step).tolist(),
                       'x3': np.linspace(parameters['x3'][0], parameters['x3'][1], step).tolist(),
                       'x4': np.linspace(parameters['x4'][0], parameters['x4'][1], step).tolist()
    }
    # 生成所有可能的参数组合
    keys, values = zip(*test_parameters.items())
    samples = [list(v) for v in itertools.product(*values)]
    return np.array(samples)


def get_test_result(file, only_latest=False):
    # 从文件中读取数据
    with open(file, 'rb') as f:
        data = pkl.load(f)

    scores = []  # 存储得分
    collisions = []  # 存储碰撞率
    for idx, scenario in data.items():
        for key, value in scenario.items():
            if key == 'final_score':
                scores.append(value)
            if key == 'collision_rate':
                collisions.append(value)

    # 将数据转换为numpy数组
    y1 = np.array(scores).reshape(-1, 1)
    y2 = np.array(collisions).reshape(-1, 1)

    if only_latest:
        # 仅返回最新的结果,用于代理模型的逐步主动采样更新
        return y1[-1,:], y2[-1,:]
    else:
        return y1, y2


def screen_most_risky_scores(init_score, exp_name, log_base_dir):
    """
    根据返回的测试场景得分，进一步分析其发生碰撞的速度和加速度信息，
    然后确定是否为真正的稀疏高危事件。

    Args:
        init_score: 初始得分
        exp_name: 实验名称，用于构造数据文件路径。如果为None，使用默认的baseline路径
        log_base_dir: 日志基础目录路径

    Returns:
        final_score: 根据速度和加速度条件筛选后的最终得分
    """
    # 根据exp_name动态构造文件路径
    records_file = f'{log_base_dir}/{exp_name}/eval_results/records.pkl'

    try:
        with open(records_file, 'rb') as f:
            batch_data = pkl.load(f)  # data是一个字典,key为场景的索引,value为该场景测试过程的list记录
            for scenario in batch_data.values():
                sequence = scenario[-1]
                ego_velocity = sequence['ego_velocity']
                min_distance2adv = sequence['min_distance']
                ego_acceleration_x = sequence['ego_acceleration_x']
                ego_acceleration_y = sequence['ego_acceleration_y']
                ego_acceleration_z = sequence['ego_acceleration_z']
                ego_acceleration = np.sqrt(ego_acceleration_x ** 2 + ego_acceleration_y ** 2 + ego_acceleration_z ** 2)
                # 根据速度和加速度来判断ego车的状态
                # for scenario 01
                if ego_velocity >= 4.0 and ego_acceleration >= 4.0:
                    final_score = init_score
                else:
                    final_score = np.array([0.031]).reshape(-1, )
        return final_score
    except FileNotFoundError:
        print(f"警告：未找到记录文件 {records_file}")
        return init_score


def estimate_IS_error(simresult, IS_weights):
    """
    估计重要性采样概率的标准差（误差水平）,不依赖真实概率

    simresult: array of 0/1 (失效事件指示)
    IS_weights: array of importance sampling weights f(x)/q(x)
    """
    n_samples = len(simresult)

    # IS 概率估计
    IS_estimates = simresult * IS_weights
    p_hat = np.mean(IS_estimates)  # 重要性采样概率估计

    # 样本方差估计
    var_hat = np.sum((IS_estimates - p_hat) ** 2) / (n_samples * (n_samples - 1))

    # 标准差（误差水平）
    sigma_hat = np.sqrt(var_hat)
    return sigma_hat, p_hat


def calculate_IS_error(samples, simresult, failre_rate, ture_prob, IS_prob):
    # 计算重要性采样的估计误差
    """
    samples: 样本数据 (n_samples, n_features)
    simresult: 仿真结果 (n_samples,)
    failre_rate: 估计的失效率
    ture_prob: 原分布下的概率 (标量)
    IS_prob: 重要性采样分布下的概率 (n_samples,)
    """
    n_samples = len(samples)

    # 计算真实分布密度 (这里需要替换为您的真实分布)
    # 注意: 在MATLAB代码中使用了mixedvinepdf，这里需要相应替换
    # 这里假设真实分布是均匀分布作为示例
    num = ture_prob
    den = IS_prob

    # 计算误差
    result = 0.0
    for i in range(n_samples):
        if simresult[i] > 0:  # 发生了关键事件
            # 计算Eq * den(i)
            den_i = failre_rate * den[i]
            # 避免除以零
            if den_i > 0:
                result += (num[i] / den_i - 1) ** 2
            else:
                result += 1  # 如果分母为零，加1作为惩罚
        else:
            # 没有发生关键事件
            result += 1

    # 计算最终误差
    error_IS = np.sqrt(result) / n_samples
    return error_IS


# 保存优化历史数据为pkl文件
def save_optimization_history(file_path, alpha_history, var_history, P_F_estimate_history):
    """保存优化历史数据"""
    history_data = {
        "alpha_history": alpha_history,
        "var_history": var_history,
        "P_F_estimate_history": P_F_estimate_history,
        "best_alpha": alpha_history[-1],
    }
    with open(file_path, "wb") as f:
        pkl.dump(history_data, f)
    print(f"Optimization history saved to {file_path}")


def plot_gmm_results(data, gmm, cluster_label, features):
    """可视化 GMM 的结果，适用于二维数据"""
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], s=10, label='Data points', alpha=0.5)

    # 绘制 GMM 的每个高斯分布
    for i, (mean, covar) in enumerate(zip(gmm.means_, gmm.covariances_)):
        # 计算协方差矩阵的特征值和特征向量
        v, w = np.linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # 95% 置信椭圆
        u = w[0] / np.linalg.norm(w[0])

        # 计算椭圆的角度
        angle = np.arctan2(u[1], u[0])
        angle = 180.0 * angle / np.pi  # 转换为角度
        ell = Ellipse(mean, v[0], v[1], 180.0 + angle, edgecolor='red', facecolor='none')
        plt.gca().add_patch(ell)

    plt.title(f'Cluster {cluster_label} - GMM Results')
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.legend()
    plt.show()


def process_clusters_with_gmm(X, labels, max_components=10, criterion='bic'):
    """
    针对每个聚类簇，使用 GMM 建模并可视化结果
    """
    gmm_models = []
    unique_labels = np.unique(labels)

    for cluster_label in unique_labels:
        cluster_data = X[labels == cluster_label]
        print(f"\nProcessing Cluster {cluster_label}...")

        # 使用 GMM 并通过 AIC/BIC 自动选择分量数
        gmm = fit_gmm_with_model_selection(cluster_data, max_components=max_components, criterion=criterion)
        gmm_models.append(gmm)
        print(f"Cluster {cluster_label}: Best number of components = {gmm.n_components}")

        # 可视化 GMM 结果（仅用二维数据展示）
        # plot_gmm_results(cluster_data, gmm, cluster_label, features)

    return gmm_models

def fit_gmm_with_model_selection(data, max_components=10, criterion='bic'):
    """
    使用 GMM 并通过 AIC 或 BIC 自动选择最佳高斯分量数
    """
    best_gmm = None
    best_score = np.inf
    for n_components in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(data)
        # 根据选择的准则计算分数
        if criterion == 'bic':
            score = gmm.bic(data)
        elif criterion == 'aic':
            score = gmm.aic(data)
        else:
            raise ValueError("Invalid criterion. Use 'aic' or 'bic'.")
        # 更新最佳模型
        if score < best_score:
            best_score = score
            best_gmm = gmm
    return best_gmm



