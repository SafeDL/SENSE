"""
step3: 将RLSearch找到的高维测试空间中的失效域进行聚类
高维测试空间中的自动驾驶失效域概率分布应该呈现出一定的多峰性, 这意味着在高维空间中可能存在多个局部最小值
但是现有研究都不能很好地捕捉到这种多峰性,本代码使用TDA拓扑分析来判定多峰性
从而确定GMM的簇个数K
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

    risky_cases = data['representative_points']
    print(f"Data Loaded: Shape {risky_cases.shape}")
    return risky_cases


def _find_multiple_eigengap(lifetimes, eta=0.5):
    """多重特征间隙法 (Multiple Eigengap)
    不再寻求单一的全局最大间隙，而是捕获所有大于最大间隙 eta 倍的显著信号间隙，
    以包容多频段的多峰结构。
    """
    if len(lifetimes) <= 1:
        return len(lifetimes) + 1
        
    gaps = lifetimes[:-1] - lifetimes[1:]
    
    # 1. 计算判定阈值
    max_gap = np.max(gaps)
    threshold_delta = eta * max_gap
    
    # 2. 提取所有显著落差的索引集合 S
    S = np.where(gaps >= threshold_delta)[0]
    
    # 3. 确定最终 K*: 取最后一个显著落差的位置
    last_significant_idx = np.max(S)
    
    # gap位于 idx 和 idx+1 之间。代表前 idx+1 个元素属于信号。
    # 额外加上1个无限相连组件，因此共有 idx+2 个聚类
    return last_significant_idx + 2


def tda_get_num_components(X, plot=True, eta=0.5):
    """
    利用持久同调(Persistent Homology)分析失效模式数量(K值)
    采用多重特征间隙法 (Multiple Eigengap) 确定最优K值

    参数:
        X: 输入数据 (N_samples, N_features)
        plot: 是否输出持久图和包含阈值划分的生命周期排序图

    返回:
        n_components: 建议的GMM组件数量
    """
    print("Running TDA to determine structural modes...")

    # 1. 维度降低
    if X.shape[1] > 50:
        pca = PCA(n_components=0.95)
        X_tda = pca.fit_transform(X)
        print(f"PCA reduced dimension to {X_tda.shape[1]} for TDA analysis.")
    else:
        X_tda = X

    # 2. 计算持久图
    result = ripser(X_tda, maxdim=0)
    diagrams = result['dgms']
    lifetimes = diagrams[0][:-1, 1]
    lifetimes = np.sort(lifetimes)[::-1]

    if len(lifetimes) <= 1:
        return len(lifetimes) + 1

    # 3. 确定 K 值 (使用多重间隙法)
    n_components = _find_multiple_eigengap(lifetimes, eta=eta)

    gaps = lifetimes[:-1] - lifetimes[1:]

    # 计算用于画横线的 Y 轴阈值（取最后一个显著断崖的两点均值作为门槛可视线）
    split_idx = n_components - 2
    if split_idx >= 0 and split_idx + 1 < len(lifetimes):
        y_threshold = (lifetimes[split_idx] + lifetimes[split_idx + 1]) / 2.0
    else:
        y_threshold = lifetimes[-1]

    print(f"TDA Analysis Result: Found {n_components} significant topological clusters (Multiple Eigengap)")
    print(f"  Top 3 gaps: {sorted(enumerate(gaps), key=lambda x: x[1], reverse=True)[:3]}")

    if plot:
        plt.figure(figsize=(12, 5))

        # 子图1: 持久图
        plt.subplot(1, 2, 1)
        plot_diagrams(diagrams, show=False)
        plt.title("Persistence Diagram (H0)")

        # 子图2: 生命周期排序图
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(lifetimes) + 1), lifetimes, 'o-', markerfacecolor='r', label='Lifetimes')
        plt.axhline(y=y_threshold, color='g', linestyle='--', linewidth=2, label=f'Multiple Eigengap Threshold (K={n_components})')
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
    file_path = '../../results/s1exp/new_search_results.pkl'
    X_data = load_data(file_path)

    # 2. TDA 分析确定组件数
    # 利用多重间隙方法自适应寻找保留的簇个数
    k_optimal = tda_get_num_components(X_data, plot=True, eta=0.5)

    # 4. 拟合 GMM
    gmm_model = fit_topological_gmm(X_data, k_optimal)

    # 5. 可视化
    visualize_gmm_results(X_data, gmm_model)

    # 6. 保存模型供 Importance Sampling 模块使用
    model_data = {
        'gmm': gmm_model,
        'tda_k': k_optimal
    }
    with open('../../results/s1exp/tda_gmm_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print("Topological GMM model saved.")


if __name__ == "__main__":
    main()