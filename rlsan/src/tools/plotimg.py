"""
数值用例需要求解的函数
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from utils import fitness_function


def visualize_results(best_positions):
    """可视化最终结果：适应度热力图 + 最优解分布 + 聚类分析"""
    # 生成网格数据 --------------------------------------------------------
    x_range = np.arange(-10, 10.1, 0.1)
    y_range = np.arange(-10, 10.1, 0.1)
    X, Y = np.meshgrid(x_range, y_range)

    # 计算适应度值（向量化计算）--------------------------------------------
    fitness_values = fitness_function((X,Y))

    print(f"min fitness value: {np.min(fitness_values)}")

    # 创建可视化画布 -----------------------------------------------------
    plt.figure(figsize=(12, 10))

    # 绘制热力图 ---------------------------------------------------------
    plt.imshow(fitness_values,
               extent=[x_range.min(), x_range.max(), y_range.min(), y_range.max()],
               cmap='hot',
               origin='lower',
               aspect='auto')
    plt.colorbar(label='Fitness Value')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Fitness Function Heatmap with Optimal Solutions')

    # 绘制最优解分布 -----------------------------------------------------
    # 假设best_positions是二维数组，形状为(N, 2)
    plt.scatter(best_positions[:, 0], best_positions[:, 1],
                c='cyan', edgecolor='black',
                label='Optimal Solutions', alpha=0.7)

    # DBSCAN聚类分析 ----------------------------------------------------
    # 注意：参数需要根据数据分布调整
    db = DBSCAN(eps=0.5, min_samples=3, metric='euclidean').fit(best_positions)
    labels = db.labels_

    # 获取聚类数量和忽略噪声点（label=-1）
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    print(f"Number of clusters: {n_clusters}")

    # 为每个聚类分配颜色
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    # 绘制聚类结果
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # 噪声点用灰色表示
            col = [0.5, 0.5, 0.5, 1]
            continue

        # 提取当前簇的数据点
        cluster_mask = (labels == k)
        cluster_points = best_positions[cluster_mask]

        # 绘制簇内点
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    c=[col], edgecolor='black',
                    label=f'Cluster {k}', alpha=0.8)

        # 计算并标注簇中心
        if len(cluster_points) > 0:
            center = np.mean(cluster_points, axis=0)
            plt.text(center[0], center[1], f'C{k}',
                     fontsize=18, weight='bold', color='white',
                     horizontalalignment='center', verticalalignment='center')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# 示例用法 -------------------------------------------------------------
if __name__ == "__main__":
    # 假设archive_positions是算法返回的最优解集合
    np.random.seed(42)
    mock_positions = np.random.uniform(-10, 10, (200, 2))  # 生成模拟数据
    visualize_results(mock_positions)