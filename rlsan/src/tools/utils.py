import matplotlib.pyplot as plt
from scipy.stats import genpareto
from ripser import Rips
import numpy as np
from joblib import Parallel, delayed
import pickle
import torch
import gpytorch
from rlsan.src.surrogate.utils import GPModel
from gpytorch.likelihoods import GaussianLikelihood
from scipy.io import savemat


class RegionModel:
    """子区域模型类,用来对应每个区域的代理模型"""
    def __init__(self, bounds):
        self.bounds = bounds
        self.X_train = None
        self.Y_train = None
        self.model = None


def save_model(sm, filename):
    """保存训练好的代理模型"""
    with open(filename, 'wb') as f:
        pickle.dump(sm, f)
    print(f"Model saved to {filename}")


def load_surrogate_model_gpu(filename):
    """加载模型参数并重建GPU版本的代理模型"""
    with open(filename, 'rb') as f:
        serialized_models = pickle.load(f)

    region_models = []
    for rm_data in serialized_models:
        # 兼容保存时有无region字段
        if 'bounds' in rm_data:
            rm = RegionModel(rm_data['bounds'])
        elif 'region' in rm_data:
            rm = RegionModel(rm_data['region'])
        else:
            raise ValueError("模型参数缺少region或bounds字段")
        rm.X_train = rm_data.get('X_train', None)
        rm.Y_train = rm_data.get('Y_train', None)

        if rm_data.get('model_state', None) is not None and rm_data.get('likelihood_state', None) is not None:
            # 重建模型和似然
            if rm.X_train is not None and rm.Y_train is not None and len(rm.X_train) > 0:
                train_x = torch.tensor(rm.X_train, dtype=torch.float32).cuda()
                train_y = torch.tensor(rm.Y_train, dtype=torch.float32).view(-1).cuda()
                likelihood = GaussianLikelihood()
                model = GPModel(train_x, train_y, likelihood).cuda()
                # 加载保存的参数
                model.load_state_dict(rm_data['model_state'])
                likelihood.load_state_dict(rm_data['likelihood_state'])
                rm.model = model
                rm.likelihood = likelihood
            else:
                rm.model = None
                rm.likelihood = None
        else:
            rm.model = None
            rm.likelihood = None

        region_models.append(rm)

    return region_models


def assign_to_regions_parallel(X, regions):
    """将训练点分配到对应的子区域,寻找处理它的区域代理模型"""
    n_features = X.shape[1]
    assignments = np.zeros(len(X), dtype=int)
    for i, region in enumerate(regions):
        mask = np.all([
            (X[:, dim] >= region[dim][0]) & (X[:, dim] <= region[dim][1])
            for dim in range(n_features)
        ], axis=0)
        assignments[mask] = i
    return assignments


def split_space(bounds, num_splits):
    """划分全局测试空间为不同子区域"""
    grids = [np.linspace(b[0], b[1], num_splits + 1) for b in bounds]
    regions = []
    from itertools import product
    for idxs in product(range(num_splits), repeat=len(bounds)):
        region = tuple((grids[i][idxs[i]], grids[i][idxs[i]+1]) for i in range(len(bounds)))
        regions.append(region)
    return regions


# def split_space(bounds, num_splits):
#     """划分全局测试空间为不同子区域"""
#     split_dims = min(3, len(bounds))
#     grids = [np.linspace(b[0], b[1], num_splits + 1) for b in bounds[:split_dims]]
#     regions = []
#     from itertools import product
#     for idxs in product(range(num_splits), repeat=split_dims):
#         region = tuple((grids[i][idxs[i]], grids[i][idxs[i]+1]) for i in range(split_dims))
#         # 其余维度直接用原始区间
#         region += tuple(bounds[i] for i in range(split_dims, len(bounds)))
#         regions.append(region)
#     return regions


def parallel_predict(region_models, region_id, X):
    """以并行的方式用子区域的Kriging模型去进行预测"""
    if region_models[region_id].model is None:
        return np.zeros(len(X)), np.ones(len(X))
    try:
        Y_pred = region_models[region_id].model.predict_values(X)
        sigma = region_models[region_id].model.predict_variances(X)
        return Y_pred.ravel(), sigma.ravel()
    except:
        return np.zeros(len(X)), np.ones(len(X))


def assign_single_point(point, regions):
    """
    预测单个测试样需要调用哪一个区域的代理模型来预测
    参数：
        point : (n_features,) 单个测试样本
        regions : 所有子区域边界列表
    返回：
        区域ID（从0开始），若未找到返回-1
    """
    point = np.array(point).reshape(1, -1)
    for region_id, region in enumerate(regions):
        in_region = all(region[i][0] <= point[0, i] <= region[i][1] for i in range(point.shape[1]))
        if in_region:
            return region_id
    return -1  # 未找到对应区域


def predict_single_sample(point, region_models, regions):
    """
    加载训练好的kriging代理模型并预测单个测试样本
    参数：
        point : (n_features,) 单个测试样本
        region_models : 所有区域模型列表
        regions : 所有子区域边界列表
    返回：
        prediction : 预测概率值
        uncertainty : 预测不确定性值
    """
    point = np.array(point).reshape(1, -1)

    # 路由到对应区域
    region_id = assign_single_point(point, regions)

    # 处理未找到区域的情况
    if region_id == -1:
        print("warning: Point not assigned to any region, returning default values.")
        return 0.0 # 默认非碰撞

    # 获取区域模型
    sm = region_models[region_id]
    prediction = sm.model.predict_values(point)
    uncertainty = sm.model.predict_variances(point)
    return prediction[0][0], uncertainty[0][0]


def predict_batch_samples(X, region_models, regions):
    """
    通过调用多核心CPU来批量预测
    参数：
        X : (n_samples, n_features) 测试样本矩阵
        region_models : 所有区域模型列表
        regions : 所有子区域边界列表
    返回：
        predictions : (n_samples,) 存储预测结果
        uncertainties : (n_samples,) 存储不确定性
    """
    # 预分配数组
    predictions = np.zeros(len(X))
    uncertainties = np.ones(len(X))

    # 批量分配区域
    assignments = assign_to_regions_parallel(X, regions)

    # 按区域并行预测
    with Parallel(n_jobs=8) as parallel:
        results = parallel(delayed(parallel_predict)(
            region_models, region_id, X[assignments == region_id]) for region_id in range(len(regions)))
        for region_id, (yp, sig2) in enumerate(results):
            mask = assignments == region_id
            predictions[mask] = yp
            uncertainties[mask] = sig2

    return predictions, uncertainties


def predict_batch_samples_gpu(X, region_models, regions):
    # 预分配数组
    Y_pred = np.zeros(len(X))
    Variance = np.ones(len(X))

    # 批量分配区域
    assignments = assign_to_regions_parallel(X, regions)

    with Parallel(n_jobs=8) as parallel:
        results = parallel(delayed(gpu_parallel_predict)(
            region_models, region_id, X[assignments == region_id]) for region_id in range(len(regions)))
        for region_id, (yp, sig2) in enumerate(results):
            mask = assignments == region_id
            Y_pred[mask] = yp
            Variance[mask] = sig2

    return Y_pred, Variance


def gpu_parallel_predict(region_models, region_id, X):
    """用子区域的GPU模型进行预测"""
    model = region_models[region_id].model
    likelihood = region_models[region_id].likelihood
    if model is None or likelihood is None or len(X) == 0:
        return np.zeros(len(X)), np.ones(len(X))
    try:
        test_x = torch.tensor(X, dtype=torch.float32).cuda()
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(model(test_x))
            mean = observed_pred.mean.cpu().numpy()
            variance = observed_pred.variance.cpu().numpy()
        return mean, variance
    except Exception:
        return np.zeros(len(X)), np.ones(len(X))


def score_to_collision_probability(score, k=10):
    """将违规得分映射为碰撞事故概率"""
    return 1 / (1 + np.exp(-k * (score - 0.5)))


def get_evt(X_data, regions, region_models, y_data):
    """拟合GPD长尾模型"""
    # 先读取拉丁采样的数据
    Y_pred = np.zeros(len(X_data))
    assignments = assign_to_regions_parallel(X_data, regions)
    with Parallel(n_jobs=8) as parallel:
        results = parallel(delayed(gpu_parallel_predict)(
            region_models, region_id, X_data[assignments == region_id]) for region_id in range(len(regions)))
        for region_id, (yp, sig2) in enumerate(results):
            mask = assignments == region_id
            Y_pred[mask] = yp

    pred_collision_probabilities = score_to_collision_probability(Y_pred)
    collision_probabilities = score_to_collision_probability(y_data)

    # 计算残差
    residuals = collision_probabilities - pred_collision_probabilities
    if np.all(residuals < 1e-6):
        print("Warning: All residuals are non-positive. EVT fitting may be invalid.")
        return None, None, None

    # 选择阈值（90%分位数）
    threshold = np.percentile(residuals, 90)
    exceedances = residuals[residuals > threshold] - threshold  # 超出阈值的部分

    # 拟合广义帕累托分布
    shape, loc, scale = genpareto.fit(exceedances, floc=0)  # 固定位置参数 loc=0

    return shape, scale, threshold


def quantum_radius(group):
    """
    用于判断例子群分布的概率云半径是否远远小于当前小生境半径，具体做法是：
    step1, 计算所有粒子的协方差矩阵,反映分布形状
    step2, 求协方差矩阵的最大特征值,代表分布的主方向方差
    step3, 对最大特征值开方,得到概率云的半径
    step4, 判断该半径是否小于小生境半径的0.1倍
    常用于进化算法中的粒子的聚集性判据
    """
    cov_matrix = np.cov(group['particles'].T)
    eigenvals = np.linalg.eigvalsh(cov_matrix)
    return np.sqrt(np.max(eigenvals)) < group['niche_radius'] * 0.1


def pheromone_decay(group, decay_rate=0.95):
    """模拟蚁群算法的信息素残留量,如果粒子发现了新解则增强信息素,否则衰减"""
    if 'pheromone' not in group:
        group['pheromone'] = 1.0

    # 每次迭代衰减，发现新解则增强
    if group['last_reward'] > np.median(group['reward_history']):
        group['pheromone'] += 0.1
    else:
        group['pheromone'] *= decay_rate

    return group['pheromone'] < 0.2


def velocity_entropy(group, window=10):
    """
    基于速度向量的信息熵判断运动活性
    如果entropy小于阈值,说明粒子的moving similarly or are inactive
    用于判断种群的活动程度
    """
    velocity_norms = np.linalg.norm(group['velocities'], axis=1)
    hist = np.histogram(velocity_norms, bins=5, range=(0,1))[0]
    prob = hist / np.sum(hist)
    entropy = -np.sum(prob * np.log(prob + 1e-10))
    return entropy < 0.5  # 经验阈值


# 计算Q值方差
def calculate_q_variance(q_table):
    all_values = []
    for key in q_table.keys():
        all_values.extend(q_table[key])  # 拼接所有值
    q_var = np.var(all_values)  # 计算方差
    return q_var


# TODO: 替换为初始建立的代理模型
# 示例适应度函数(多峰函数)
def fitness_function(position):
    x, y = position
    return 10 * (np.sin(x) * np.cos(y)) + (x ** 2 + y ** 2) / 10


# 示例适应度函数(单峰函数)
# def fitness_function(position):
#     x, y = position
#     dip = -5 * np.exp(-((x - 2) ** 2 + (y - 2) ** 2) / 5)  # 局部最小值在 (2, 2)
#     global_positive = 1  # 确保其他区域为正值
#     return global_positive + dip


# 示例适应度函数(双峰函数)
# def fitness_function(position):
#     x, y = position
#     # 定义两个正值峰
#     peak1 = 3 * np.exp(-((x - 5) ** 2 + (y - 5) ** 2) / 10)  # 尖峰在 (5, 5)
#     peak2 = 2 * np.exp(-((x + 5) ** 2 + (y + 5) ** 2) / 20)  # 宽峰在 (-5, -5)
#     # 定义两个负值的局部最小值区域
#     dip1 = -4 * np.exp(-((x - 2) ** 2 + (y - 2) ** 2) / 5)  # 局部最小值在 (2, 2)
#     dip2 = -3 * np.exp(-((x + 3) ** 2 + (y + 3) ** 2) / 8)  # 局部最小值在 (-3, -3)
#     # 组合所有部分
#     return peak1 + peak2 + dip1 + dip2


def compute_topological_features(positions):
    """基于持续同调（persistent homology）提取拓扑特征"""
    rips = Rips(verbose=False)  # 一个来自 ripser 库的类，用于计算 Vietoris-Rips 复形的持久同调
    diagrams = rips.fit_transform(positions)

    # 算给定位置的持久同调，并返回持久条码图
    h1 = diagrams[1]
    if len(h1) > 0:
        persistence = np.max(h1[:, 1] - h1[:, 0])
        birth_time = np.mean(h1[:, 0])
    else:
        persistence = 0.0
        birth_time = 0.0

    # 返回一个包含持久性和出生时间的数组
    # persistence: 表示子种群在搜索空间中形成的拓扑结构（如连通分量或环）存在的时间长度,持久性越长，表示该拓扑结构越稳定(可能对应于子种群在搜索空间空间中的某个稳定区域)
    # birth_time: 表示拓扑结构形成的时间,出生时间越早，表示该拓扑结构越早形成(可能对应于子种群在搜索空间中的某个重要区域)

    return np.array([persistence, birth_time])


def update_archive(subgroups):
    """更新存档"""
    archive_positions_all = []
    archive_fitness_all = []
    for group in subgroups:
        archive_positions_all.append(group['archive_positions'])
        archive_fitness_all.append(group['archive_fitness'])
    # 将fitness为负数的找出来,并且删除
    archive_positions_all = [pos[fit < 0] for pos, fit in zip(archive_positions_all, archive_fitness_all)]
    return np.vstack(archive_positions_all), np.hstack(archive_fitness_all)


def visualize_learning(rewards, q_variances, current_iter):
    """可视化学习曲线"""
    plt.style.use('classic')

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(rewards[:current_iter], label='Average Reward')
    plt.title('Learning Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Reward')
    plt.grid()
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(q_variances[:current_iter], label='Q-value Variance', color='g')
    plt.xlabel('Iterations')
    plt.ylabel('Q-value Variance')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()


def visualize_f1_scores(f1_scores):
    # 假设 f1_scores 是从 main_vfe_parallel 函数中获取的 F1 分数列表
    # f1_scores = [...]  # 这里是你的 F1 分数数据

    # 设置全局字体和风格
    plt.style.use('seaborn')  # 使用 seaborn 风格，适合学术论文
    plt.rcParams['font.family'] = 'Times New Roman'  # 设置字体为 Times New Roman，符合学术规范
    plt.rcParams['font.size'] = 12  # 设置字体大小
    plt.rcParams['figure.figsize'] = (8, 6)  # 设置图表尺寸，适合论文排版
    plt.rcParams['figure.dpi'] = 300  # 设置分辨率为 300 DPI，确保图像清晰

    # 创建图表
    fig, ax = plt.subplots()

    # 绘制 F1 分数曲线
    iterations = np.arange(len(f1_scores))
    ax.plot(iterations, f1_scores,
            color='dodgerblue',  # 使用专业化的蓝色
            linewidth=2.5,  # 线条粗细
            marker='o',  # 添加圆点标记
            markersize=6,  # 标记大小
            markeredgecolor='white',  # 标记边缘颜色
            markeredgewidth=1.5,  # 标记边缘宽度
            label='F1 Score')  # 添加图例标签

    # 设置标题和标签
    ax.set_xlabel('Iteration', fontsize=12)  # x 轴标签
    ax.set_ylabel('F1 Score', fontsize=12)  # y 轴标签

    # 设置坐标轴范围（可选，根据数据调整）
    ax.set_xlim(0, iterations[-1] + 2)  # 稍微扩展 x 轴范围以避免边界裁剪
    ax.set_ylim(0, 1.05)  # F1 分数范围通常为 0 到 1，稍微扩展 y 轴上限

    # 添加网格
    ax.grid(True,
            linestyle='--',  # 网格线样式
            alpha=0.7,  # 网格线透明度
            color='gray')  # 网格线颜色

    # 设置刻度样式
    ax.tick_params(axis='both',
                   which='major',
                   labelsize=10,
                   direction='in',  # 刻度朝内
                   length=5)  # 刻度线长度

    # 添加图例
    ax.legend(loc='lower right',  # 图例位置
              fontsize=10,
              frameon=True,  # 显示图例边框
              framealpha=0.9,  # 边框透明度
              edgecolor='black')  # 边框颜色

    # 调整布局以避免标签被裁剪
    plt.tight_layout()

    # 保存图像为高分辨率文件，适合出版
    # plt.savefig('f1_score_progression.pdf',
    #             format='pdf',  # 使用 PDF 格式，适合学术出版
    #             bbox_inches='tight',  # 确保不裁剪内容
    #             dpi=300)  # 高分辨率

    # 显示图像
    plt.show()


def save_data_to_mat(mean_rewards, std_rewards):
    """保存数据到 .mat 文件"""
    data_to_save = {
        'mean_rewards': np.array(mean_rewards),      # 每次迭代的平均 reward
        'std_rewards': np.array(std_rewards),        # 每次迭代的 reward 标准差
    }

    # 保存到 .mat 文件
    savemat('reward_data.mat', data_to_save)
    print("数据已保存到 reward_data.mat 文件中")