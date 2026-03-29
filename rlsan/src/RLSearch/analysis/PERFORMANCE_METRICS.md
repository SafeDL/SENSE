# RLSAN 搜索性能指标说明

## 性能评估指标概述

为了更全面地评估自动驾驶测试搜索算法的性能，我们在 `run_rlsan_search.py` 中添加了两个重要的性能指标。

---

## 📊 指标详解

### 1. AUC-FDC (Area Under Curve - Failure Discovery Curve)

#### 定义
- **横轴**：仿真次数（Simulation Budget）
- **纵轴**：累计失效用例数（Cumulative Number of Failures Found）
- **计算方法**：使用梯形法则（trapezoidal rule）计算曲线下面积

#### 含义
- **越大越好**：AUC 越大，说明算法在相同仿真预算下发现的失效越多，**早期搜索效率越高**
- 对比两个算法：如果曲线更早"抬头"（快速发现失效），AUC 就更大
- 这个指标反映了**搜索效率**的核心性能

#### 数学形式
```
AUC-FDC = ∫ (failures) d(budget)
        = Σ [(f_i + f_{i+1})/2 × (b_i+1 - b_i)]
```

#### 示例
```
算法A的FDC曲线：快速上升 → AUC 大（高效）
算法B的FDC曲线：缓慢上升 → AUC 小（低效）
```

---

### 2. N₅₀ (Budget to find 50 failures)

#### 定义
- 算法发现**前 50 个失效用例**所需的**仿真步数**
- 如果算法找不到 50 个失效，则 N₅₀ = -1

#### 含义
- **越小越好**：N₅₀ 越小，说明算法找到前 50 个失效用时越少，搜索效率越高
- 相对于"第一个失效"的指标（time_to_first_failure）更稳定
- **有效抹平随机性**：
  - 第一个失效可能带有运气成分
  - 前 50 个失效能够客观反映算法的**持续搜索能力**

#### 优势
1. **稳定性更好**：避免因幸运发现第一个失效而带来的偏差
2. **全局衡量**：反映算法的整体性能，不只看开始
3. **对比友好**：两个算法的 N₅₀ 更容易比较

#### 示例
```
算法A：N_50 = 500   （用 500 次仿真找到 50 个失效）
算法B：N_50 = 800   （用 800 次仿真找到 50 个失效）
→ 算法A 性能更好
```

---

## 📈 输出示例

运行 `run_rlsan_search.py` 后，你会看到类似的输出：

```
Optimization completed in 123.45 seconds.
Total Carla Calls: 1000
Hazardous scenarios found: 87
Time to First Failure (Budget): 42

--- Performance Metrics ---
AUC-FDC (Failure Discovery Curve): 3.2145e+04
N_50 (Budget to find 50 failures): 256
```

---

## 💾 保存数据

所有指标都被保存在 `search_results.pkl` 中：

```python
results = {
    'hazardous_points': np.array([...]),        # 所有找到的危险点
    'time_to_first_failure': 42,                # 第一个失效的仿真步数
    'total_simulations': 1000,                  # 总仿真次数
    'search_time': 123.45,                      # 总搜索时间（秒）
    'auc_fdc': 3.2145e+04,                      # ★ 新增指标
    'n_50': 256,                                # ★ 新增指标
    'fdc_curve': [(0,0), (10,1), (50,5), ...], # FDC 曲线所有点
}
```

---

## 🔍 加载和分析结果

```python
import pickle
import numpy as np
import matplotlib.pyplot as plt

# 加载结果
with open('search_results.pkl', 'rb') as f:
    results = pickle.load(f)

# 访问指标
auc_fdc = results['auc_fdc']
n_50 = results['n_50']
fdc_curve = results['fdc_curve']

# 绘制 FDC 曲线
budgets = [p[0] for p in fdc_curve]
failures = [p[1] for p in fdc_curve]

plt.figure(figsize=(10, 6))
plt.plot(budgets, failures, marker='o', label='FDC Curve')
plt.axhline(y=50, color='r', linestyle='--', label='N=50')
plt.axvline(x=n_50, color='g', linestyle='--', label=f'N_50={n_50}')
plt.xlabel('Simulation Budget')
plt.ylabel('Cumulative Failures')
plt.title(f'Failure Discovery Curve (AUC-FDC = {auc_fdc:.4e})')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 📋 对比分析

### 指标对比表

| 指标 | 衡量维度 | 优势 | 适用场景 |
|-----|--------|-----|--------|
| **time_to_first_failure** | 首次发现效率 | 简单直观 | 快速评估 |
| **AUC-FDC** | 早期搜索效率 | 全局衡量 | 详细分析 |
| **N_50** | 持续搜索能力 | 稳定可靠 | 算法对比 |

### 如何使用三个指标

1. **快速筛选**：看 `time_to_first_failure` 判断算法是否有效
2. **效率评估**：用 `AUC-FDC` 衡量早期搜索能力
3. **性能对比**：用 `N_50` 对比不同算法的稳定性

---

## ⚠️ 注意事项

1. **最小仿真次数**：如果总仿真次数少于 50，N₅₀ 会是 -1
2. **FDC 曲线**：如果需要绘制曲线，可以使用 `results['fdc_curve']` 的数据
3. **梯形法则**：AUC-FDC 使用梯形法则计算，精度依赖于采样密度

---

## 📚 参考

- 梯形法则：`numpy.trapz(y, x)`
- 原论文参考：Failure Discovery Curve (FDC) 来自测试有效性评估的最佳实践

