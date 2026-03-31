"""
验证重要性采样方法相比较于传统的蒙特卡罗方法,在寻找测试空间中罕见的失效事件的优势,
特别是评估IS方法能否发现MC难以探测的盲区
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

def load_data(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def filter_failures(X, y, threshold=0.3):
    """提取经过物理过滤后的失效点"""
    y = np.array(y).flatten()
    mask = y > threshold
    return X[mask]

# ================= 配置路径 =================
# 注意：请确保 y_is 和 y_mc 都是已经经过“速度+加速度”过滤后的真实物理分数
is_params_file = '../../results/IS/scenario05/scenario05_20251228/sampled_parameters.pkl'
is_scores_file = '../../results/s5exp/IS_simulation_scores_20251228.pkl'

mc_params_file = '../../results/MCMC/scenario05/sampled_parameters.pkl'
mc_scores_file = '../../results/s5exp/MCMC_simulation_scores.pkl'

# 1. 加载数据
X_is = load_data(is_params_file)
Y_is = load_data(is_scores_file)
X_mc = load_data(mc_params_file)[:80000] # 保持与你之前的分析量一致
Y_mc = load_data(mc_scores_file)[:80000]

# 2. 提取失效点点云
is_failures = filter_failures(X_is, Y_is)
mc_failures = filter_failures(X_mc, Y_mc)

print(f"IS Found Failure Points: {len(is_failures)}")
print(f"MC Found Failure Points: {len(mc_failures)}")

# 3. --- 核心分析：最近邻距离分析 (Nearest Neighbor Distance) ---
# 计算 80,000 个 MC 样本在 [-1, 1]^3 空间中的理论平均间距
# 空间体积 V=8, N=80000, 密度 rho = 10000 samples/unit_vol.
# 平均间距约为 (V/N)^(1/3) = 0.046
avg_mc_spacing = (8.0 / 80000)**(1/3)
blind_threshold = 2.0 * avg_mc_spacing # 定义 2 倍间距外为“盲区”

# 使用 NN 搜索
nn = NearestNeighbors(n_neighbors=1).fit(X_mc)
distances, indices = nn.kneighbors(is_failures)
distances = distances.flatten()

# 4. 计算漏检指标
blind_mask = distances > blind_threshold
blind_points = is_failures[blind_mask]
discovery_rate = len(blind_points) / len(is_failures)

print(f"\n--- Topological Discovery Analysis ---")
print(f"Theoretical MC Grid Spacing: {avg_mc_spacing:.4f}")
print(f"Blind Spot Threshold: {blind_threshold:.4f}")
print(f"Failures in MC-Blind Regions: {len(blind_points)} / {len(is_failures)}")
print(f"New Region Discovery Rate: {discovery_rate:.2%}")

# 5. --- 模式识别：IS 到底多找出了几个“坑”？ ---
# 对盲区点进行聚类，看它们属于几个不同的失效模式
if len(blind_points) > 0:
    db = DBSCAN(eps=0.15, min_samples=10).fit(blind_points)
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    print(f"Number of NEW failure modes found by IS: {n_clusters}")

# 6. --- 可视化对比 ---
fig = plt.figure(figsize=(12, 6))

# 左图：MC 捕捉到的失效空间
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(mc_failures[:,0], mc_failures[:,1], mc_failures[:,2], c='blue', alpha=0.3, s=2, label='MC Failures')
ax1.set_title("Failures Captured by MC\n(Limited Visibility)")
ax1.set_xlim([-1, 1]); ax1.set_ylim([-1, 1]); ax1.set_zlim([-1, 1])

# 右图：IS 捕捉到的新增失效空间
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(is_failures[~blind_mask, 0], is_failures[~blind_mask, 1], is_failures[~blind_mask, 2],
           c='gray', alpha=0.2, s=2, label='Common Failures')
ax2.scatter(blind_points[:,0], blind_points[:,1], blind_points[:,2],
           c='red', alpha=0.6, s=10, label='IS Newly Discovered')
ax2.set_title(f"Failures Discovered by IS\n({discovery_rate:.1%} New Regions)")
ax2.set_xlim([-1, 1]); ax2.set_ylim([-1, 1]); ax2.set_zlim([-1, 1])
ax2.legend()

plt.tight_layout()
plt.show()