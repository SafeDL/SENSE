import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from scipy.spatial import ConvexHull


class DiversityEvaluator:
    def __init__(self, min_samples=10):
        """
        :param min_samples: 形成核心簇所需的最小样本数 (保持不变，代表对密度的最低要求)
        """
        self.min_samples = min_samples

    def auto_tune_dbscan(self, data):
        """
        自适应寻找最佳 eps 参数
        目标：最大化 (Silhouette Score) 同时 最小化 (Outlier Ratio)
        """
        # 定义搜索范围：从 0.05 到 0.5，步长 0.01
        eps_candidates = np.arange(0.05, 0.52, 0.01)

        best_score = -2.0  # 初始最低分
        best_model = None
        best_labels = None
        best_eps = 0.1
        best_n_clusters = 0
        best_s_score = -1

        print(f"  > Auto-tuning eps in range [0.05, 0.5]...")

        for eps in eps_candidates:
            clustering = DBSCAN(eps=eps, min_samples=self.min_samples).fit(data)
            labels = clustering.labels_

            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)
            n_clusters = len(unique_labels)

            # 只有当发现 2 个以上簇时才计算有效分数
            if n_clusters < 2:
                continue

            # 1. 计算轮廓系数 (Quality)
            valid_mask = labels != -1
            if np.sum(valid_mask) > n_clusters:
                s_score = silhouette_score(data[valid_mask], labels[valid_mask])
            else:
                s_score = -1

            # 2. 计算离群值比例 (Quantity Loss)
            n_noise = np.sum(labels == -1)
            n_total = len(labels)
            outlier_ratio = n_noise / n_total

            # 3. 综合评分公式
            # Score = S_Score - (Penalty * Outlier_Ratio)
            # 这里的 0.5 是惩罚系数(越大则越痛恨离群值),意味着我们愿意牺牲一点点 S-Score 来换取更少的离群值
            composite_score = s_score - (0.5 * outlier_ratio)

            if composite_score > best_score:
                best_score = composite_score
                best_model = clustering
                best_labels = labels
                best_eps = eps
                best_n_clusters = n_clusters
                best_s_score = s_score

        if best_model is None:
            print("  ! Warning: Auto-tune failed to find valid clusters. Fallback to eps=0.2")
            clustering = DBSCAN(eps=0.2, min_samples=self.min_samples).fit(data)
            return clustering.labels_, 0.2, -1, 0

        print(f"  > Best eps found: {best_eps:.2f} (S-Score: {best_s_score:.2f}, Clusters: {best_n_clusters})")
        return best_labels, best_eps, best_s_score, best_n_clusters

    def evaluate(self, hazardous_cases, save_path='diversity_analysis.png', title_prefix=''):
        if len(hazardous_cases) < self.min_samples:
            print("Not enough cases for diversity analysis.")
            return None

        # === 1. 执行自适应参数搜索 ===
        labels, optimal_eps, s_score, n_clusters = self.auto_tune_dbscan(hazardous_cases)

        # === 2. 计算覆盖广度 (Global Spread) ===
        try:
            if hazardous_cases.shape[1] > 1 and len(hazardous_cases) > hazardous_cases.shape[1] + 1:
                global_volume = ConvexHull(hazardous_cases).volume
            else:
                global_volume = 0
        except:
            global_volume = 0

        print(f"[{title_prefix}] Final Metrics:")
        print(f"  > Modes Found: {n_clusters}")
        print(f"  > Silhouette Score: {s_score:.4f}")
        print(f"  > Global Volume: {global_volume:.4f}")
        print(f"  > Optimal Eps Used: {optimal_eps:.2f}")

        # === 3. IEEE 风格绘图 ===
        self._plot_clusters(hazardous_cases, labels, n_clusters, s_score, save_path, title_prefix)

        return {
            'n_clusters': n_clusters,
            'silhouette_score': s_score,
            'coverage': global_volume
        }

    def _plot_clusters(self, data, labels, n_clusters, s_score, save_path, title_prefix):
        # IEEE 格式设置
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['font.size'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9

        fig, ax = plt.subplots(figsize=(3.5, 3.5))

        # 颜色映射 (固定随机种子以保持颜色一致性)
        unique_labels = set(labels)
        sorted_labels = sorted(list(unique_labels))
        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(sorted_labels))]

        # 统计噪声比例，用于标题显示
        outlier_count = np.sum(labels == -1)
        outlier_pct = (outlier_count / len(labels)) * 100

        for k, col in zip(sorted_labels, colors):
            class_member_mask = (labels == k)
            xy = data[class_member_mask]

            if k == -1:
                # 噪声点：黑色/灰色，无边框
                c_noise = [0.5, 0.5, 0.5, 0.5]  # 半透明灰色
                ax.scatter(xy[:, 0], xy[:, 1], s=8, c=[c_noise], marker='x',
                           alpha=0.4, linewidth=0.5, label='Outliers')
            else:
                # 核心簇：有黑色边框
                ax.scatter(xy[:, 0], xy[:, 1], s=25, c=[col], marker='o',
                           alpha=0.9, linewidth=0.5, edgecolor='k', label=f'Mode {k + 1}')

        # 标题包含更多信息
        ax.set_title(f'{title_prefix}\n(S-Score: {s_score:.2f}, Modes: {n_clusters}, Outliers: {outlier_pct:.1f}%)',
                     fontsize=9)
        ax.set_xlabel('Latent Dim 1')
        ax.set_ylabel('Latent Dim 2')
        ax.grid(True, linestyle='--', alpha=0.4)

        # 智能图例
        handles, leg_labels = ax.get_legend_handles_labels()
        if 'Outliers' in leg_labels:
            out_idx = leg_labels.index('Outliers')
            out_h, out_l = handles.pop(out_idx), leg_labels.pop(out_idx)
            handles.append(out_h)
            leg_labels.append(out_l)

        if len(handles) > 6:
            ax.legend(handles[:6], leg_labels[:6], loc='best', fontsize=7, frameon=True)
        else:
            ax.legend(handles, leg_labels, loc='best', fontsize=7, frameon=True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        print(f"Diversity plot saved to {save_path}")