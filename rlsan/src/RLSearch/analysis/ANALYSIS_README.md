# RLSAN 搜索结果分析脚本

本文件夹中包含用于分析和可视化 RLSAN 搜索结果的脚本。

## 📋 脚本说明

### 1. `quick_view.py` - 快速查看数据
在终端中快速查看关键指标，无需生成图表。

**使用方法：**
```bash
python scripts/quick_view.py
python scripts/quick_view.py --results_path log/search_results.pkl
```

**输出内容：**
- 搜索摘要（总评估数、CARLA仿真数、搜索时间）
- 原始失效指标（总数、AUC-FDC、N_50）
- 代表性失效指标（总数、AUC-FDC、N_50、去重率）
- FDC曲线采样点展示

**优点：**
- ✅ 无需安装 matplotlib，快速运行
- ✅ 在任何终端环境中都能使用
- ✅ 适合快速检查结果

---

### 2. `analyze_search_results.py` - 完整分析与可视化
生成详细的分析报告和可视化图表。

**使用方法：**
```bash
python scripts/analyze_search_results.py
python scripts/analyze_search_results.py --results_path log/search_results.pkl --output_dir results
```

**生成的输出：**

#### 命令行输出
- 详细的摘要统计信息
- 所有关键指标的打印输出

#### 生成的文件（保存在 `results/` 目录）

1. **fdc_curves.png** - Failure Discovery Curve 对比
   - 左图：原始失效的FDC曲线
   - 右图：代表性失效的FDC曲线
   - 包含AUC-FDC和N_50标注

2. **comparison_metrics.png** - 性能指标对比（4个子图）
   - 左上：总失效数对比（柱状图）
   - 右上：AUC-FDC值对比（柱状图）
   - 左下：N_50值对比（柱状图）
   - 右下：评估预算分布（饼状图）

3. **convergence_analysis.png** - 收敛性分析
   - 左图：失效发现率随时间的变化
   - 右图：累积失效曲线对比

4. **analysis_report.txt** - 完整文本报告
   - 基础统计信息
   - 原始和代表性失效指标
   - 效率指标计算
   - 各指标的解释说明

**参数说明：**
- `--results_path`: search_results.pkl 文件路径（默认：`log/search_results.pkl`）
- `--output_dir`: 输出目录（默认：`results/`）

---

### 3. `analyze.sh` - 便捷脚本
一行命令运行完整分析（自动安装依赖）。

**使用方法：**
```bash
bash scripts/analyze.sh
bash scripts/analyze.sh log/search_results.pkl results
```

**功能：**
- ✅ 自动检测并安装 numpy、matplotlib
- ✅ 调用 `analyze_search_results.py`
- ✅ 适合在Linux/Mac环境中使用

---

## 🚀 快速开始

### 方案 1：快速查看（推荐先用这个）
```bash
python scripts/quick_view.py
```
1行命令，3秒内查看所有关键指标。

### 方案 2：生成完整报告
```bash
python scripts/analyze_search_results.py
```
生成4张图表和1份详细报告（需要 numpy 和 matplotlib）。

### 方案 3：一键运行（自动安装依赖）
```bash
bash scripts/analyze.sh
```
适合CI/CD流程或自动化分析。

---

## 📊 关键指标解释

### AUC-FDC (Area Under Curve - Failure Discovery Curve)
- **定义**：Failure Discovery Curve 下的面积
- **横轴**：仿真/评估次数
- **纵轴**：累计失效用例数
- **衡量**：早期搜索效率
- **评价**：越大越好 ✓

**含义示例：**
```
算法A: AUC = 5.2e+04  →  高效（快速发现失效）
算法B: AUC = 2.1e+04  →  低效（缓慢发现失效）
```

### N_50 (Budget to find 50 failures)
- **定义**：发现前50个失效用例所需的仿真步数
- **值域**：正整数或 -1（未达到）
- **衡量**：持续搜索能力
- **评价**：越小越好 ✓

**含义示例：**
```
算法A: N_50 = 500   →  用500步找到50个失效
算法B: N_50 = 1000  →  用1000步找到50个失效
→ 算法A性能更好
```

### Raw vs Representative（原始 vs 代表性）

| 指标 | 含义 | 用途 |
|------|------|------|
| **Raw Failures** | 所有发现的失效（包含相似的） | 衡量发现的总数量 |
| **Representative** | 去重后的失效（具有多样性） | 衡量发现的质量和多样性 |

**去重率：** `Raw / Representative`
- 去重率高 = 发现的失效聚集在少数几个模式中
- 去重率低 = 发现的失效分散在不同模式中

---

## 🔍 分析结果示例

### 快速查看输出
```
======================================================================
RLSAN SEARCH RESULTS - QUICK VIEW
======================================================================

[1] SEARCH SUMMARY
  Total Evaluations:        42100
  Real CARLA Simulations:   1203
  Search Time:              1234.56 s

[2] RAW FAILURES
  Total Found:              123
  AUC-FDC:                  6.35e+07
  N_50:                     8543

[3] REPRESENTATIVE FAILURES
  Total Found:              45
  AUC-FDC:                  2.89e+07
  N_50:                     12456
  Deduplication Ratio:      2.73x

[4] RAW FAILURES FDC CURVE (sample)
  Budget    Failures
       0          0
     4200          12
     8400          28
    12600          45
    ...
```

### 完整分析输出
生成4张精美图表：
- `fdc_curves.png`: 两条FDC曲线对比
- `comparison_metrics.png`: 4个关键指标的可视化对比
- `convergence_analysis.png`: 收敛速度分析
- `analysis_report.txt`: 详细文本报告

---

## ⚙️ 依赖要求

### `quick_view.py`
- Python 3.6+
- 无额外依赖

### `analyze_search_results.py`
- Python 3.6+
- numpy（用于数组操作）
- matplotlib（用于绘图）

### 自动安装
```bash
pip install numpy matplotlib
# 或
pip3 install numpy matplotlib
```

---

## 📝 常见问题

### Q: 为什么 N_50_Representative 这么大？
**A:** 这是正常的！原因是：
- Raw Failures 包含大量相似的失效（聚集在同一失效模式）
- Representative Failures 是经过去重后的（具有多样性）
- 找到50个**不同的**失效模式比找到50个**任意**失效要花费更长时间

### Q: 去重率很高意味着什么？
**A:**
- **高去重率**（如 5.0x）：算法发现的失效聚集在少数几个模式，多样性较低
- **低去重率**（如 1.5x）：算法发现的失效分散在多个模式，多样性较高

### Q: 如何对比两次运行的结果？
**A:**
1. 保存两个 search_results.pkl 到不同目录
2. 分别运行分析脚本
3. 对比生成的图表和报告中的指标

### Q: 图表保存在哪里？
**A:** 默认保存在 `results/` 目录（相对于当前工作目录）
- 可通过 `--output_dir` 参数指定

---

## 💡 使用建议

1. **第一次运行**：先用 `quick_view.py` 快速查看数据
2. **详细分析**：用 `analyze_search_results.py` 生成图表和报告
3. **对比实验**：保存多个 search_results.pkl，分别分析后对比
4. **论文发表**：使用生成的图表和报告数据

---

## 📚 相关文档

- [PERFORMANCE_METRICS.md](../PERFORMANCE_METRICS.md) - 详细的性能指标说明
- [run_rlsan_search.py](./run_rlsan_search.py) - 搜索脚本（生成 search_results.pkl）

---

更新时间：2026-03-24
