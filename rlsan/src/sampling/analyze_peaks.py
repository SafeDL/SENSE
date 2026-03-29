"""
step3: 将RLSearch找到的高维测试空间中的失效域进行聚类
高维测试空间中的自动驾驶失效域概率分布应该呈现出一定的多峰性, 这意味着在高维空间中可能存在多个局部最小值
但是现有研究都不能很好地捕捉到这种多峰性,本代码使用TDA拓扑分析来判定多峰性
从而确定GMM的簇个数K以及每个簇的初始均值位置
"""
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
# 忽略特定警告
warnings.filterwarnings("ignore")


def load_data(filepath):
    """加载数据"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Data Loaded: Shape {data.shape}")
    return data


def tda_get_num_components(X, plot=True, decay_threshold=0.5):
    """
    利用持久同调(Persistent Homology)分析失效模式数量(K值)

    参数:
        X: 输入数据 (N_samples, N_features)
        decay_threshold:用于区分信号和噪声的阈值比例。
                        (生命周期 < max_lifetime * threshold 的被视为噪声)
    返回:
        n_components: 建议的GMM组件数量
        initial_means: 基于拓扑聚类的粗略中心估计 (可选)
    """
    print("Running TDA to determine structural modes...")

    # 1. 如果维度过高(>50), 先用PCA降噪, 保留主要方差, 避免高维稀疏性影响TDA
    if X.shape[1] > 50:
        pca = PCA(n_components=0.95)  # 保留95%方差
        X_tda = pca.fit_transform(X)
        print(f"PCA reduced dimension to {X_tda.shape[1]} for TDA analysis.")
    else:
        X_tda = X

    # 2. 为了加速计算，如果样本量极大，可采用Landmark采样(此处假设样本量<5000)
    # 计算持久图 (maxdim=0 关注连通分量 H0)
    # ripser 计算 H0 非常快
    result = ripser(X_tda, maxdim=0)
    diagrams = result['dgms']

    # 提取 H0 特征 (birth, death)
    # H0 的 birth 都是 0, 我们只关心 death time (即生命周期 lifetime)
    # 最后一个分量的 death 是 inf，代表整体连通性
    lifetimes = diagrams[0][:-1, 1]
    lifetimes = np.sort(lifetimes)[::-1]  # 降序排列

    # 3. 确定 K 值 (寻找寿命骤降点 - Eigengap heuristic)
    # 方法：找到最大的 gap，或者保留生命周期显著长的分量
    if len(lifetimes) == 0:
        return 1

    # 简单策略：保留生命周期大于最大生命周期 10% 的分量
    max_life = lifetimes[0]
    significant_indices = np.where(lifetimes > max_life * decay_threshold)[0]
    n_components = len(significant_indices) + 1  # +1 是因为包含那个 infinite 的主分量

    print(f"TDA Analysis Result: Found {n_components} significant topological clusters.")

    if plot:
        plt.figure(figsize=(10, 5))

        # 子图1: 持久图
        plt.subplot(1, 2, 1)
        plot_diagrams(diagrams, show=False)
        plt.title("Persistence Diagram (H0)")

        # 子图2: 生命周期排序图 (Scree Plot 风格)
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(lifetimes) + 1), lifetimes, 'o-', markerfacecolor='r')
        plt.axhline(y=max_life * decay_threshold, color='g', linestyle='--', label='Noise Threshold')
        plt.xlabel("Component Rank")
        plt.ylabel("Lifetime (Persistence)")
        plt.title("Topological 'Scree Plot'")
        plt.legend()
        plt.tight_layout()
        plt.savefig('../../results/s1exp/tda_analysis.png')
        plt.show()

    return n_components


def fit_topological_gmm(X, n_components):
    """
    基于TDA建议的K值拟合GMM
    """
    print(f"Fitting GMM with K={n_components} based on TDA prior...")

    # 初始化 GMM
    # n_init=5 表示虽然我们指定了K，但为了稳健还是多跑几次EM
    gmm = GaussianMixture(n_components=n_components,
                          covariance_type='full',
                          reg_covar=1e-5,  # 增加正则化，防止奇异矩阵
                          n_init=10,
                          random_state=42,
                          init_params='kmeans')  # 这里可以用 kmeans++ 初始化，效果通常很好

    gmm.fit(X)

    # 输出 BIC 以供参考 (虽然我们主要信赖 TDA)
    print(f"GMM Converged. BIC: {gmm.bic(X):.2f}")
    return gmm


def visualize_gmm_results(X, gmm):
    """可视化 GMM 聚类结果 (降维到2D展示)"""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    labels = gmm.predict(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=10, alpha=0.6)

    # 绘制粗略的中心
    centers_pca = pca.transform(gmm.means_)
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=100, marker='X', label='GMM Centers')

    plt.colorbar(scatter, label='Cluster ID')
    plt.title(f"GMM Clustering (K={gmm.n_components}) Visualization")
    plt.legend()
    plt.savefig('../../results/s1exp/gmm_result.png')
    plt.show()


def main():
    # 1. 加载 RL 搜索出的失效样本
    file_path = '../../results/s1exp/rl_searched_cases.pkl'
    X_data = load_data(file_path)

    # 2. TDA 分析确定组件数
    # decay_threshold 越小，保留的簇越多（越敏感）
    k_optimal = tda_get_num_components(X_data, plot=True, decay_threshold=0.5)

    # 4. 拟合 GMM
    gmm_model = fit_topological_gmm(X_data, k_optimal)

    # 5. 可视化
    visualize_gmm_results(X_data, gmm_model)

    # 6. 保存模型供 Importance Sampling 模块使用
    # 注意：保存时最好同时也保存 scaler, 因为预测新数据需要同样的 scaling
    model_data = {
        'gmm': gmm_model,
        'tda_k': k_optimal
    }
    with open('../../results/s1exp/tda_gmm_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print("Topological GMM model saved.")


if __name__ == "__main__":
    main()