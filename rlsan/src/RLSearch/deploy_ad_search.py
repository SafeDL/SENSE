"""
deploy_ad_search.py - 危险工况搜索的辅助函数
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')


def select_representative_seeds(hazardous_points: np.ndarray,
                                fitness_values: np.ndarray,
                                niche_radius: float = 0.05,
                                niche_capacity: int = 3) -> dict:
    """
    筛选最危险和最有代表性的危险种子

    Args:
        hazardous_points: 危险点集合 (N, dim)
        fitness_values: 对应的适应度值 (N,)
        niche_radius: 小生境半径
        niche_capacity: 每个小生境容量

    Returns:
        dict 包含代表性种子及其适应度
    """
    if len(hazardous_points) == 0:
        return {
            'representative': np.array([]),
            'representative_fitness': np.array([])
        }

    # 去重
    _, unique_idx = np.unique(np.round(hazardous_points, decimals=4), axis=0, return_index=True)
    unique_points = hazardous_points[unique_idx]
    unique_fitness = fitness_values[unique_idx]

    # 基于小生境覆盖的代表性种子选择
    sorted_idx = np.argsort(unique_fitness)
    n_unique = len(unique_points)
    coverage_counts = np.zeros(n_unique)
    used = np.zeros(n_unique, dtype=bool)
    representative = []
    representative_fitness = []

    for i in sorted_idx:
        if used[i] or coverage_counts[i] >= niche_capacity:
            continue

        representative.append(unique_points[i])
        representative_fitness.append(unique_fitness[i])
        used[i] = True

        distances = np.linalg.norm(unique_points - unique_points[i], axis=1)
        coverage_counts[distances < niche_radius] += 1

    representative = np.array(representative) if representative else np.array([])
    representative_fitness = np.array(representative_fitness) if representative_fitness else np.array([])

    return {
        'representative': representative,
        'representative_fitness': representative_fitness,
    }


def plot_seeds(seeds: np.ndarray, fitness: np.ndarray, title: str, save_path: str):
    """
    可视化种子分布

    Args:
        seeds: 种子点 (N, dim)
        fitness: 适应度值
        title: 图表标题
        save_path: 保存路径
    """
    if len(seeds) == 0:
        print(f"  No seeds to plot for {title}")
        return

    dim = seeds.shape[1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. 2D 散点图
    if dim >= 2:
        scatter = axes[0].scatter(seeds[:, 0], seeds[:, 1],
                                   c=fitness, cmap='RdYlGn_r',
                                   s=60, edgecolors='black', linewidths=0.5)
        plt.colorbar(scatter, ax=axes[0], label='Fitness')
        axes[0].set_xlabel('Dim 1')
        axes[0].set_ylabel('Dim 2')
        axes[0].set_title(f'{title} (N={len(seeds)})')
        axes[0].grid(True, linestyle='--', alpha=0.3)

    # 2. 适应度分布
    axes[1].hist(fitness, bins=min(20, len(fitness)), color='coral',
                 alpha=0.7, edgecolor='black')
    axes[1].axvline(np.min(fitness), color='red', linestyle='--',
                    label=f'Min: {np.min(fitness):.4f}')
    axes[1].set_xlabel('Fitness')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Fitness Distribution')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.3)

    # 3. 各维度分布
    for d in range(min(dim, 5)):
        axes[2].hist(seeds[:, d], bins=20, alpha=0.5, label=f'Dim {d+1}')
    axes[2].set_xlabel('Value')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Parameter Distribution')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved: {save_path}")
