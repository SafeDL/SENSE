# RLSAN 分析工具使用指南

本文档介绍如何使用 RLSAN 分析工具进行搜索结果的分析和可视化，包括新增的失效域覆盖率(FDC)指标。

## 📂 文件结构

```
analysis/
├── analyze_search_results.py    # 完整分析脚本（包含FDC）
├── quick_view.py                # 快速查看脚本（包含FDC）
├── ANALYSIS_README.md           # 详细文档
└── results/                      # 输出目录
    ├── fdc_curves.png
    ├── comparison_metrics.png
    ├── convergence_analysis.png
    └── analysis_report.txt
```

## 🚀 快速开始

### 1. 快速查看（推荐先用）

```bash
cd /home/hp/SENSE/rlsan/src/RLSearch/analysis

# 不包含覆盖率
python3 quick_view.py --results_path ../../../../log/search_results.pkl

# 包含覆盖率（需要grid数据）
python3 quick_view.py \
  --results_path ../../../../log/search_results.pkl \
  --grid_x ../../surrogate/train_data/scenario01_grid_x.pkl \
  --grid_y ../../surrogate/train_data/scenario01_grid_y.pkl
```

### 2. 完整分析（生成图表和报告）

```bash
# 生成基础分析（不含覆盖率）
python3 analyze_search_results.py \
  --results_path ../../../../log/search_results.pkl \
  --output_dir results

# 生成完整分析（包含覆盖率）
python3 analyze_search_results.py \
  --results_path ../../../../log/search_results.pkl \
  --output_dir results \
  --grid_x ../../surrogate/train_data/scenario01_grid_x.pkl \
  --grid_y ../../surrogate/train_data/scenario01_grid_y.pkl
```

### 3. 自定义碰撞阈值

```bash
# 如果grid_y的碰撞判定阈值不是0.3，可以自定义
python3 analyze_search_results.py \
  --results_path ../../../../log/search_results.pkl \
  --grid_x ../../surrogate/train_data/scenario01_grid_x.pkl \
  --grid_y ../../surrogate/train_data/scenario01_grid_y.pkl \
  --collision_threshold 0.5  # 自定义阈值
```

---

## 📊 输出指标说明

### 基础指标

1. **AUC-FDC (绝对值)**
   - 公式：∫ failures d(budget)
   - 含义：失效发现曲线下的面积
   - 评价：越大越好

2. **nAUC-FDC (归一化)**
   - 公式：nAUC = AUC / (T_total × N_max)
   - 含义：每单位预算的失效发现效率
   - 评价：越大越好

3. **N_50**
   - 含义：发现前50个失效所需的预算
   - 评价：越小越好

### 新增指标：失效域覆盖率（FDC）

#### 定义

将测试空间 χ ∈ ℝ³ 划分为 M = 30×30×30 = 27,000 个不相交的网格胞腔。

**Ground Truth (基准集合):**
$$\mathcal{C}_{fail} = \{ c \in \mathcal{C} \mid \exists x_{gs} \in X_{GS}, x_{gs} \in c \wedge I_f(x_{gs})=1 \}$$

**Captured Set (捕获集合):**
$$\mathcal{C}_{RLSAN} = \{ c \in \mathcal{C}_{fail} \mid \exists x_{found} \in X_{found}, x_{found} \in c \}$$

**覆盖率:**
$$CRate(X_{found}) = \frac{|\mathcal{C}_{RLSAN}|}{|\mathcal{C}_{fail}|} \times 100\%$$

#### 实现步骤

1. **空间量化**
   - 网格分辨率：30×30×30（每维30个点）
   - 范围：[-1, 1] × [-1, 1] × [-1, 1]
   - 步长：δ = 2 / (30 - 1) ≈ 0.0690

2. **建立基准**
   - 加载 scenario01_grid_y.pkl
   - 标记所有 y > 0.3 的点为碰撞
   - 这些点映射到的网格单元集合为 C_fail
   - 本示例：|C_fail| = 97 个胞腔

3. **计算覆盖**
   - 对 RLSAN 找到的每个危险点，映射到网格
   - 检查该网格是否在 C_fail 中
   - 统计被覆盖的胞腔数量
   - 本示例：|C_RLSAN| = 24 个胞腔

4. **输出结果**
   - 覆盖率 = 24 / 97 = 24.74%

#### 物理含义

- **高覆盖率（>80%）**: RLSAN 发现的失效分布广泛，覆盖了大部分危险区域
- **中等覆盖率（50-80%）**: RLSAN 发现了一半以上的危险区域
- **低覆盖率（<50%）**: RLSAN 倾向于聚焦在少数几个危险区域，多样性不足

#### 对比示例

| 场景 | Raw Failures | Representative | Coverage Rate | 解释 |
|------|-------------|----------------|----------------|------|
| 聚集式搜索 | 500 | 10 | 10% | 找到很多相似的失效，但覆盖范围小 |
| 多样性强 | 100 | 80 | 80% | 找到的失效分散在广泛的区域 |
| 平衡搜索 | 200 | 40 | 40% | 介于两者之间 |

---

## 📈 输出文件说明

### analysis_report.txt

完整的文本报告，包含：
- 基础统计
- 原始失效指标
- 代表性失效指标
- 效率指标
- 失效域覆盖率
- 各指标解释

### fdc_curves.png

两条失效发现曲线对比：
- 左图：原始失效 FDC 曲线
- 右图：代表性失效 FDC 曲线
- 都标注了 AUC-FDC 和 N_50

### comparison_metrics.png

6 个子图的综合对比：
1. 总失效数（柱状图）
2. 绝对 AUC-FDC（柱状图）
3. **归一化 nAUC-FDC（柱状图）**← 新增
4. N_50 预算（柱状图）
5. 评估预算分布（饼图）
6. **效率指标汇总（文本框）**← 更新

### convergence_analysis.png

收敛性分析：
- 左图：失效发现率变化
- 右图：累积失效曲线对比

---

## 💾 数据来源

### 搜索结果
- **位置**: `/home/hp/SENSE/log/search_results.pkl`
- **内容**:
  - `hazardous_points`: RLSAN 发现的危险点
  - `auc_fdc_raw / auc_fdc_representative`: 曲线下面积
  - `n_50_raw / n_50_representative`: 预算指标
  - `fdc_curve_raw / fdc_curve_representative`: 完整的 FDC 曲线数据

### Grid Search 基准
- **X 坐标**: `/home/hp/SENSE/rlsan/src/surrogate/train_data/scenario01_grid_x.pkl`
  - 形状: (27000, 3)
  - 内容: 30×30×30 网格的坐标点

- **Y 值**: `/home/hp/SENSE/rlsan/src/surrogate/train_data/scenario01_grid_y.pkl`
  - 长度: 27000
  - 内容: 每个点的风险值

---

## 🔧 参数说明

### analyze_search_results.py

```bash
usage: analyze_search_results.py [-h] [--results_path RESULTS_PATH]
                                 [--output_dir OUTPUT_DIR]
                                 [--grid_x GRID_X]
                                 [--grid_y GRID_Y]
                                 [--collision_threshold COLLISION_THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  --results_path RESULTS_PATH
                        Path to search_results.pkl (default: ../log/search_results.pkl)
  --output_dir OUTPUT_DIR
                        Directory to save plots (default: results)
  --grid_x GRID_X       Path to scenario01_grid_x.pkl (optional)
  --grid_y GRID_Y       Path to scenario01_grid_y.pkl (optional)
  --collision_threshold COLLISION_THRESHOLD
                        Collision detection threshold (default: 0.3)
```

### quick_view.py

```bash
usage: quick_view.py [-h] [--results_path RESULTS_PATH]
                     [--grid_x GRID_X]
                     [--grid_y GRID_Y]
                     [--collision_threshold COLLISION_THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  --results_path RESULTS_PATH
                        Path to search_results.pkl
  --grid_x GRID_X       Path to grid X coordinates (optional)
  --grid_y GRID_Y       Path to grid Y values (optional)
  --collision_threshold COLLISION_THRESHOLD
                        Collision threshold (default: 0.3)
```

---

## 📝 典型工作流程

### 场景 1: 快速评估（2分钟）

```bash
# 终端命令
python3 quick_view.py --results_path search_results.pkl

# 输出: 基本统计 + FDC曲线采样
```

### 场景 2: 详细分析 + 覆盖率（5分钟）

```bash
# 终端命令
python3 quick_view.py \
  --results_path search_results.pkl \
  --grid_x grid_x.pkl \
  --grid_y grid_y.pkl

# 然后运行
python3 analyze_search_results.py \
  --results_path search_results.pkl \
  --grid_x grid_x.pkl \
  --grid_y grid_y.pkl \
  --output_dir my_results

# 输出：txt报告 + 3张图表
```

### 场景 3: 论文发表（对比多个实验）

```bash
# 实验A
python3 analyze_search_results.py \
  --results_path exp_A/search_results.pkl \
  --output_dir exp_A_results \
  --grid_x grid_x.pkl --grid_y grid_y.pkl

# 实验B
python3 analyze_search_results.py \
  --results_path exp_B/search_results.pkl \
  --output_dir exp_B_results \
  --grid_x grid_x.pkl --grid_y grid_y.pkl

# 对比 exp_A_results/ 和 exp_B_results/ 中的数据和图表
```

---

## ⚠️ 常见问题

### Q: 为什么覆盖率这么低（<30%）？

**A:** 这是正常现象。原因：
- Grid search 识别的 97 个危险胞腔分散在整个空间
- RLSAN 的 791 个危险点虽然数量多，但聚集在少数几个胞腔
- 这反映了 RLSAN 的"深度优先"策略 vs Grid Search 的"广泛搜索"

### Q: 覆盖率和 "Representative Failures" 的关系？

**A:**
- **Representative = 104**: RLSAN 找到的 104 个不同的失效簇（去重后）
- **Coverage = 24.74%**: 这 104 个簇覆盖了 Grid Search 的 97 个胞腔中的 24 个
- 两者衡量的维度不同：
  - Representative 衡量多样性（在 RLSAN 内部去重）
  - Coverage 衡量覆盖面（相对于 Grid Search 真值）

### Q: 如果没有 Grid Search 数据怎么办？

**A:**
- 只能运行不含 `--grid_x` 和 `--grid_y` 的分析
- 可以使用 AUC-FDC、nAUC-FDC、N_50 等指标评估
- Coverage Rate 会显示为 0

### Q: 碰撞阈值 0.3 是否可以调整？

**A:** 可以，但需要与 Grid Search 使用的阈值一致
```bash
python3 analyze_search_results.py \
  --collision_threshold 0.5  # 如果 Grid Search 用的是 0.5
```

---

## 📚 学术参考

### Failure Domain Coverage 公式

$$CRate = \frac{|\{c \in C_{fail} : \exists x \in X_{found}, x \in c\}|}{|C_{fail}|} \times 100\%$$

### 网格索引映射

对于点 $x = [x_1, x_2, x_3]$：

$$Idx(x) = \lfloor \frac{x - x_{min}}{\Delta} \rfloor$$

其中 $\Delta = \frac{x_{max} - x_{min}}{n_{grid} - 1}$

### 去重机制

使用 `select_representative_seeds()` 函数进行聚类去重，距离阈值为 `niche_radius = 0.05`

---

## 📞 技术支持

遇到问题？检查：

1. **文件路径是否正确** → 使用 `ls` 验证
2. **依赖库是否安装** → 运行 `pip install numpy matplotlib`
3. **Python 版本** → 需要 3.6+
4. **数据格式** → Grid 数据必须是 .pkl 格式

---

更新时间：2026-03-24
