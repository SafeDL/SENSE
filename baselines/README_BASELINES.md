# 基线方法使用说明

## 概述

本目录实现了6种搜索基线方法，全部集成了代理模型加速功能，用于与RLSAN进行公平的性能对比。

## 特性

- **统一接口**：所有基线方法使用 `SurrogateEvaluator` 包装器
- **代理模型加速**：支持使用预训练的高斯过程模型快速评估
- **自动结果格式化**：输出与RLSAN兼容的格式
- **统计追踪**：自动分离代理调用和真实CARLA仿真次数

## 实现的基线方法

### 1. 随机搜索 (Random Search)
**文件**: `random_search.py`

均匀随机采样测试空间，不使用任何优化策略。

```bash
python random_search.py \
  --budget 1000 \
  --use_surrogate True \
  --output_dir log/baselines
```

**参数**:
- `--budget`: 总评估次数预算 (默认: 10)
- `--use_surrogate`: 是否使用代理模型 (默认: True)

**输出**: `random_search_results.pkl`

---

### 2. 遗传算法 (Genetic Algorithm)
**文件**: `genetic_search.py`

基于种群的进化算法，通过选择、交叉和变异操作进化种群。

```bash
python genetic_search.py \
  --budget 1000 \
  --ga_pop_size 50 \
  --use_surrogate True \
  --output_dir log/baselines
```

**参数**:
- `--budget`: 总评估次数预算 (默认: 1000)
- `--ga_pop_size`: 种群大小 (默认: 50)
- `--use_surrogate`: 是否使用代理模型 (默认: True)

**输出**: `genetic_algorithm_results.pkl`

---

### 3. 贝叶斯优化 (Bayesian Optimization)
**文件**: `bayesian_optimization.py`

使用高斯过程模型和LCB采集函数的顺序优化方法。

```bash
python bayesian_optimization.py \
  --budget 1000 \
  --use_surrogate True \
  --output_dir log/baselines
```

**参数**:
- `--budget`: 总评估次数预算 (默认: 1000)
- `--use_surrogate`: 是否使用代理模型 (默认: True)

**输出**: `bayesian_optimization_results.pkl`

---

### 4. 随机邻域搜索 (Random Neighbourhood Search)
**文件**: `random_neighbourhood_search.py`

Epsilon-Greedy策略: 在全局随机探索和局部开发之间权衡。

```bash
python random_neighbourhood_search.py \
  --budget 1000 \
  --rnns_explore_rate 0.3 \
  --rnns_perturb 0.1 \
  --use_surrogate True \
  --output_dir log/baselines
```

**参数**:
- `--budget`: 总评估次数预算 (默认: 1000)
- `--rnns_explore_rate`: 随机探索概率 (默认: 0.3)
- `--rnns_perturb`: 局部扰动标度 (默认: 0.1)
- `--use_surrogate`: 是否使用代理模型 (默认: True)

**输出**: `random_neighbourhood_search_results.pkl`

---

### 5. 斥力自适应采样 (Repulsive Adaptive Sampling)
**文件**: `repulsive_adaptive_sampling.py`

基于Ge et al. 2024论文的方法，使用物理斥力场引导采样。

```bash
python repulsive_adaptive_sampling.py \
  --budget 1000 \
  --batch_size 10 \
  --use_surrogate True \
  --output_dir log/baselines
```

**参数**:
- `--budget`: 总评估次数预算 (默认: 1000)
- `--batch_size`: 批处理大小 (默认: 10)
- `--use_surrogate`: 是否使用代理模型 (默认: True)

**输出**: `repulsive_adaptive_sampling_results.pkl`

---

### 6. Learning to Collide (LC)
**文件**: `learning_to_collide.py`

基于REINFORCE策略梯度的对抗性初始状态生成方法。使用训练好的LC模型（自回归高斯策略网络）生成对抗性初始位置参数。

```bash
python learning_to_collide.py \
  --budget 2000 \
  --lc_model_id scenario_05_weight \
  --output_dir log/baselines
```

**参数**:
- `--budget`: 总评估次数预算 (默认: 2000000)
- `--lc_model_dir`: LC模型权重目录 (默认: `safebench/scenario/scenario_data/model_ckpt/lc`)
- `--lc_model_id`: 模型文件名前缀 (默认: `scenario_05_weight`)
- `--lc_standard_action_dim`: True=4维动作, False=3维动作 (默认: True)

**输出**: `learning_to_collide_results.pkl`

---

## 代理模型配置

### 模型加载

所有脚本默认从以下路径加载代理模型:
```
../rlsan/results/s1exp/surrogate_model_1000.pkl
```

### 模型格式

代理模型为pickle文件，包含:
```python
{
    'model': gpytorch.models.ExactGP,  # 高斯过程模型
    'likelihood': gpytorch.likelihoods.GaussianLikelihood  # 噪声模型
}
```

### 使用自定义模型

```bash
python random_search.py \
  --surrogate_model_path /path/to/custom_model.pkl \
  --budget 1000
```

### 禁用代理模型（使用真实CARLA）

```bash
python random_search.py \
  --use_surrogate False \
  --budget 1000
```

---

## 结果文件格式

所有基线方法都输出RLSAN兼容的格式 (pickle字典):

```python
{
    'method': 'Random Search',           # 方法名称
    'algorithm': 'baseline',
    'hazardous_points': np.array(...),   # 找到的危险点 (N, 3)
    'all_samples': np.array(...),        # 所有评估的样本 (M, 3)
    'all_scores': np.array(...),         # 所有评估的分数 (M,)
    'raw_failures_count': 123,           # 危险点总数
    'representative_failures_count': 123,# 去重后的危险点
    'total_evaluations': 1000,           # 总评估次数
    'real_simulations': 1000,            # 真实CARLA仿真次数
    'surrogate_calls': 0,                # 代理模型调用次数
    'search_time': 42.5,                 # 搜索耗时 (秒)
    'coverage_rate_raw': 25.7,           # 覆盖率 (%)
    'coverage_rate_rep': 25.7,
    'captured_cells_raw': 25,            # 捕获的网格点数
    'captured_cells_rep': 25,
    'ground_truth_cells': 97,            # 真值网格点数
}
```

---

## 数据对比分析

### 运行所有基线（推荐）

创建脚本 `run_all_baselines.sh`:

```bash
#!/bin/bash

BUDGET=1000
OUTPUT_DIR="log/baselines"
SURROGATE_PATH="../rlsan/results/s1exp/surrogate_model_1000.pkl"

mkdir -p $OUTPUT_DIR

echo "[*] Running Random Search..."
python random_search.py --budget $BUDGET --output_dir $OUTPUT_DIR --surrogate_model_path $SURROGATE_PATH

echo "[*] Running Genetic Algorithm..."
python genetic_search.py --budget $BUDGET --output_dir $OUTPUT_DIR --surrogate_model_path $SURROGATE_PATH

echo "[*] Running Bayesian Optimization..."
python bayesian_optimization.py --budget $BUDGET --output_dir $OUTPUT_DIR --surrogate_model_path $SURROGATE_PATH

echo "[*] Running Random Neighbourhood Search..."
python random_neighbourhood_search.py --budget $BUDGET --output_dir $OUTPUT_DIR --surrogate_model_path $SURROGATE_PATH

echo "[*] Running Repulsive Adaptive Sampling..."
python repulsive_adaptive_sampling.py --budget $BUDGET --output_dir $OUTPUT_DIR --surrogate_model_path $SURROGATE_PATH

echo "[✓] All baselines completed!"
```

执行:
```bash
cd baselines
bash run_all_baselines.sh
```

### 统一对比分析

```bash
cd baselines
python analyze_unified_baseline_results.py \
  --baselines_dir ../log/baselines \
  --rlsan_results ../log/search_results.pkl \
  --output_dir ../results/baseline_comparison
```

**输出文件**:
- `baseline_comparison.png`: 4个子图对比
- `baseline_comparison_report.txt`: 详细文本报告

---

## 性能指标

### 关键指标

| 指标 | 说明 |
|------|------|
| Hazardous Found | 发现的危险测试场景数 |
| Coverage Rate | 覆盖Grid Search危险区域的百分比 |
| Search Time | 总搜索耗时（秒） |
| Total Evals | 代理+真实评估次数总和 |
| Real Sims | 真实CARLA仿真次数 |
| Surrogate | 代理模型调用次数 |

### 加速效果评估

代理模型加速的效果通过以下指标衡量:

```
加速倍数 = Total Evals / Real Sims
```

- `1.0`: 未使用代理（全真实CARLA）
- `> 1.0`: 使用了代理加速

---

## 故障排除

### 代理模型加载失败

```
[!] Surrogate model not found at ...
```

**解决方案**:
1. 检查模型文件是否存在
2. 指定正确的路径: `--surrogate_model_path /path/to/model.pkl`
3. 或禁用代理: `--use_surrogate False`

### CARLA连接错误

确保CARLA服务器正在运行:
```bash
cd CARLA_ROOT
./CarlaUE4.sh --world-port=2000 -opengl
```

### 预算分配不当

**问题**: 遗传算法的实际评估次数 ≠ 指定的预算

**原因**: GA的代数 = 预算 / 种群大小

**解决方案**: 调整种群大小或预算
```bash
# 例如: 预算1000，种群50 → 20代，总1000次评估
python genetic_search.py --budget 1000 --ga_pop_size 50
```

---

## 论文写作建议

### 相关工作对比

```markdown
**代理模型加速**：所有基线方法都采用相同的预训练高斯过程模型
（1000个训练数据点），确保公平的性能对比。代理模型调用
极快（<1ms/点），主要评估预算用于真实CARLA仿真。

**评估预算统一**：所有方法（包括RLSAN）使用相同的评估
预算上限，确保横向对比的科学性。
```

### 实验结果展示

```markdown
表X：不同搜索方法的性能对比

| 方法 | 危险点 | 覆盖率 | 耗时 | 实际仿真 |
|------|-------|--------|------|---------|
| 随机搜索 | 45 | 12.4% | 1500s | 1000 |
| 遗传算法 | 67 | 18.2% | 1520s | 1000 |
| 贝叶斯优化 | 89 | 24.7% | 1510s | 1000 |
| RLSAN | 156 | 52.6% | 950s | 1000 |
```

---

## 参考文献

- Bayesian Optimization: [scikit-optimize](https://scikit-optimize.github.io/)
- Genetic Algorithm: 标准GA实现
- Random Neighbourhood: Epsilon-Greedy exploration-exploitation
- Adaptive Sampling: Ge et al., "Life-long Learning and Testing", TIV 2024

---

**最后更新**: 2026-03-24
