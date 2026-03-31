"""
可视化IS替代分布在grid测试结果上的概率密度
"""
import pickle
import os.path as osp
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from scipy.special import logsumexp

matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']


def load_is_package(package_path):
    """加载IS包（包含GMM、alpha、Z_GMM）"""
    print(f"[*] Loading IS package from {package_path}")
    with open(package_path, 'rb') as f:
        package = pickle.load(f)

    gmm = package.get('gmm')
    alpha = package.get('lambda_mix', package.get('alpha', 0.1))
    Z_GMM = package.get('Z_constant', package.get('Z_GMM', 1.0))
    print(f"    GMM components: {gmm.n_components}, alpha: {alpha}, Z_GMM: {Z_GMM}")
    return gmm, alpha, Z_GMM


def load_grid_data(grid_x_path, grid_y_path):
    """加载grid测试数据"""
    print(f"[*] Loading grid X from {grid_x_path}")
    with open(grid_x_path, 'rb') as f:
        grid_x = pickle.load(f)

    print(f"[*] Loading grid Y from {grid_y_path}")
    with open(grid_y_path, 'rb') as f:
        grid_y = pickle.load(f)

    return np.array(grid_x), np.array(grid_y)


def compute_q_mix_density(samples, gmm, alpha, Z_GMM):
    """计算防御性混合密度"""
    log_q_gmm = gmm.score_samples(samples) - np.log(Z_GMM)

    dim = samples.shape[1]
    vol = 2.0 ** dim
    log_q_uniform = np.full(samples.shape[0], -np.log(vol))

    alpha = np.clip(alpha, 1e-10, 1 - 1e-10)
    log_prob_gmm = np.log(1 - alpha) + log_q_gmm
    log_prob_def = np.log(alpha) + log_q_uniform

    log_q_mix = logsumexp(np.vstack([log_prob_gmm, log_prob_def]).T, axis=1)
    return np.exp(log_q_mix)


def plot_density_heatmap(gmm, alpha, Z_GMM, grid_x, failure_mask, output_dir):
    """绘制IS密度的2D平面投影热力图 (叠加红X危险场景)"""
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    dim = gmm.means_.shape[1]

    if dim >= 3:
        # 绘制前三个参数的两两截面组合 (其他维度默认补0)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        pairs = [(0, 1, 'Param 1', 'Param 2'), 
                 (0, 2, 'Param 1', 'Param 3'), 
                 (1, 2, 'Param 2', 'Param 3')]
        
        for ax, (idx1, idx2, label1, label2) in zip(axes, pairs):
            samples = np.zeros((X.size, dim))
            samples[:, idx1] = X.ravel()
            samples[:, idx2] = Y.ravel()
            
            q_mix = compute_q_mix_density(samples, gmm, alpha, Z_GMM)
            Z = q_mix.reshape(X.shape)
            
            contourf = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
            ax.contour(X, Y, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)

            # 在背景用红XX绘制危险测试场景
            failures_x = grid_x[failure_mask, idx1]
            failures_y = grid_x[failure_mask, idx2]
            ax.scatter(failures_x, failures_y, c='red', marker='x', s=15, alpha=0.8, label='Failures')

            ax.set_xlabel(label1, fontsize=12)
            ax.set_ylabel(label2, fontsize=12)
            ax.set_title(f'{label1} vs {label2} (α={alpha:.2f})', fontsize=12)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.legend(loc='upper right')
            plt.colorbar(contourf, ax=ax, label='Probability Density')
            
        plt.tight_layout()
        output_path = osp.join(output_dir, 'is_density_heatmap_projections.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[✓] Saved heatmaps to {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_package', type=str,
                       default='/home/hp/SENSE/rlsan/results/s1exp/final_refined_IS_package.pkl',
                       help='Path to IS package')
    parser.add_argument('--grid_x', type=str,
                       default='/home/hp/SENSE/rlsan/src/surrogate/train_data/scenario01_grid_x.pkl',
                       help='Path to grid X')
    parser.add_argument('--grid_y', type=str,
                       default='/home/hp/SENSE/rlsan/src/surrogate/train_data/scenario01_grid_y.pkl',
                       help='Path to grid Y')
    parser.add_argument('--output_dir', type=str, default='results/is_density_viz',
                       help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    gmm, alpha, Z_GMM = load_is_package(args.is_package)
    grid_x, grid_y = load_grid_data(args.grid_x, args.grid_y)

    print(f"[*] Grid shape: {grid_x.shape}, Grid Y shape: {grid_y.shape}")
    print(f"[*] Z_GMM interpretation: {Z_GMM:.4f} = {100*Z_GMM:.1f}% of GMM mass in [-1,1]^D")

    # 识别失效区域
    failure_mask = grid_y > 0.3
    n_failures = np.sum(failure_mask)
    print(f"[*] Failure points: {n_failures}/{len(grid_y)}")

    # 绘制热力图并包含危险场景点位
    plot_density_heatmap(gmm, alpha, Z_GMM, grid_x, failure_mask, args.output_dir)

    # 计算IS密度
    q_mix = compute_q_mix_density(grid_x, gmm, alpha, Z_GMM)

    # 绘制对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    scatter1 = axes[0].scatter(grid_x[:, 0], grid_x[:, 1], c=grid_y, cmap='RdYlGn_r', s=20, alpha=0.6)
    axes[0].set_title('Grid Test Results (Y values)')
    axes[0].set_xlabel('Param 1')
    axes[0].set_ylabel('Param 2')
    plt.colorbar(scatter1, ax=axes[0])

    scatter2 = axes[1].scatter(grid_x[:, 0], grid_x[:, 1], c=q_mix, cmap='viridis', s=20, alpha=0.6)
    axes[1].set_title(f'IS Mixture Density (α={alpha:.2f})')
    axes[1].set_xlabel('Param 1')
    axes[1].set_ylabel('Param 2')
    plt.colorbar(scatter2, ax=axes[1])

    q_mix_on_failures = q_mix[failure_mask]
    scatter3 = axes[2].scatter(grid_x[failure_mask, 0], grid_x[failure_mask, 1],
                               c=q_mix_on_failures, cmap='plasma', s=30, alpha=0.7)
    axes[2].set_title(f'IS Density on Failures (n={n_failures})')
    axes[2].set_xlabel('Param 1')
    axes[2].set_ylabel('Param 2')
    plt.colorbar(scatter3, ax=axes[2])

    plt.tight_layout()
    output_path = osp.join(args.output_dir, 'is_density_on_grid.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[✓] Saved to {output_path}")

    print(f"\n[Statistics]")
    print(f"  IS density on failures: mean={np.mean(q_mix_on_failures):.6f}, "
          f"std={np.std(q_mix_on_failures):.6f}")
    print(f"  IS density on safe: mean={np.mean(q_mix[~failure_mask]):.6f}, "
          f"std={np.std(q_mix[~failure_mask]):.6f}")
    print(f"  Ratio (failure/safe): {np.mean(q_mix_on_failures) / np.mean(q_mix[~failure_mask]):.2f}x")


if __name__ == '__main__':
    main()
