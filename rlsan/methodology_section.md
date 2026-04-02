# 方法论 (Methodology)

## 2. 方法论

### 2.1 问题建模

自动驾驶系统的边界情况搜索可建模为多模态优化问题。设搜索空间为 $\mathcal{X} \subseteq \mathbb{R}^d$，目标函数为 $f: \mathcal{X} \rightarrow \mathbb{R}$，其中 $f(\mathbf{x})$ 表示场景 $\mathbf{x}$ 的安全性评分（负值表示危险）。

**定义 1（危险场景）**：场景 $\mathbf{x}$ 为危险场景，当且仅当 $f(\mathbf{x}) < \tau_d$，其中 $\tau_d$ 为危险阈值。

**定义 2（小生境覆盖率）**：将搜索空间划分为网格单元 $\mathcal{G} = \{g_1, g_2, \ldots, g_m\}$，覆盖率定义为：
$$\text{Coverage} = \frac{|\{g_i : \exists \mathbf{x} \in g_i, f(\mathbf{x}) < \tau_d\}|}{m}$$

### 2.2 状态感知小生境粒子群优化 (State-Aware Niche PSO)

#### 2.2.1 架构概述

提出的方法整合了四个核心组件：

1. **进化状态分类器**：自动识别搜索阶段
2. **状态感知动作空间**：简化的离散/连续动作集
3. **聚类多样性奖励**：密集学习信号
4. **增强NDM算子**：跳出局部最优

#### 2.2.2 进化状态分类

使用四分类器识别搜索阶段：

$$\text{State} = \begin{cases}
\text{Exploration} & \text{if } \text{Diversity} > \theta_d \text{ and } \text{Stagnation} < \theta_s \\
\text{Exploitation} & \text{if } \text{Diversity} \leq \theta_d \text{ and } \text{Improvement} > \theta_i \\
\text{Stagnation} & \text{if } \text{Stagnation} \geq \theta_s \\
\text{Restart} & \text{otherwise}
\end{cases}$$

其中：
- $\text{Diversity} = \frac{1}{n}\sum_{i=1}^{n} \|\mathbf{x}_i - \bar{\mathbf{x}}\|_2$ 为种群多样性
- $\text{Stagnation}$ 为无改进迭代次数
- $\text{Improvement} = \frac{f_{\text{prev}} - f_{\text{curr}}}{|f_{\text{prev}}|}$ 为适应度改进率

#### 2.2.3 增强状态特征表示 (28维)

设第 $g$ 个子种群的状态向量为 $\mathbf{s}_g \in \mathbb{R}^{28}$，由以下特征组成：

**特征组 1：进化状态 One-Hot (4维)**
$$\mathbf{s}^{(1)}_g = \text{OneHot}(\text{State}_g) \in \{0,1\}^4$$

**特征组 2：适应度统计 + 正弦嵌入 (8维)**

变异系数（无量纲化）：
$$CV_f = \tanh\left(\frac{\sigma_f}{|\mu_f| + \epsilon}\right)$$

其中 $\mu_f, \sigma_f$ 分别为子种群适应度的均值和标准差。

正弦嵌入（频率带数 $K=2$）：
$$\mathbf{e}_{\sin}(x) = [\sin(\pi x), \cos(\pi x), \sin(2\pi x), \cos(2\pi x)]^T$$

$$\mathbf{s}^{(2)}_g = [\mathbf{e}_{\sin}(CV_f); \mathbf{e}_{\sin}(\text{Improvement}_{\text{norm}})]$$

**特征组 3：多样性特征 (4维)**

位置多样性（归一化）：
$$D_{\text{pos}} = \frac{1}{d}\sum_{j=1}^{d} \text{std}(\mathbf{x}_{:,j}^{\text{norm}})$$

其中 $\mathbf{x}^{\text{norm}} = \frac{\mathbf{x} - \mathbf{x}_{\min}}{\mathbf{x}_{\max} - \mathbf{x}_{\min}}$

速度比：
$$R_v = \frac{\text{mean}(\|\mathbf{v}_i\|_2)}{v_{\max} \sqrt{d}}$$

聚集度：
$$C = \frac{\text{mean}(\|\mathbf{x}_i - \bar{\mathbf{x}}\|_2)}{\sqrt{d}}$$

$$\mathbf{s}^{(3)}_g = [D_{\text{pos}}, R_v, C, \text{radius\_ratio}]^T$$

**特征组 4：时间嵌入 (5维)**

$$\mathbf{s}^{(4)}_g = [p, \sin(2\pi p), \cos(2\pi p), \mathbb{1}_{p<0.3}, \mathbb{1}_{p>0.7}]^T$$

其中 $p = \frac{t}{T_{\max}}$ 为搜索进度。

**特征组 5：交互特征 (4维)**

子种群间距离（归一化）：
$$d_{\min} = \frac{\min_{i \neq g} \|\mathbf{g}^*_i - \mathbf{g}^*_g\|_2}{\sqrt{d}}$$

$$\mathbf{s}^{(5)}_g = [d_{\min}, d_{\text{mean}}, \text{overlap}, \text{isolated}]^T$$

**特征组 6：年龄特征 (1维)**

$$\text{age}_g = \tanh\left(\frac{\text{iterations\_since\_restart}}{20}\right)$$

**特征组 7：子种群感知特征 (3维)**

排名（相对位置）：
$$\text{rank}_g = \frac{\text{argsort}(f^*_g)}{N_g - 1}$$

相对适应度：
$$f_{\text{rel}} = \frac{f^*_g - f^*_{\min}}{f^*_{\max} - f^*_{\min}}$$

危险区域标志：
$$\text{danger\_flag} = \mathbb{1}_{f^*_g < \tau_d}$$

$$\mathbf{s}^{(7)}_g = [\text{rank}_g, f_{\text{rel}}, \text{danger\_flag}]^T$$

**完整状态向量**：
$$\mathbf{s}_g = [\mathbf{s}^{(1)}_g; \mathbf{s}^{(2)}_g; \mathbf{s}^{(3)}_g; \mathbf{s}^{(4)}_g; \mathbf{s}^{(5)}_g; \mathbf{s}^{(6)}_g; \mathbf{s}^{(7)}_g] \in \mathbb{R}^{28}$$

### 2.3 Soft Actor-Critic (SAC) 强化学习控制

#### 2.3.1 SAC算法框架

SAC是基于最大熵强化学习的离策算法，目标是最大化期望累积奖励与策略熵的加权和：

$$J(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{T} \gamma^t (r_t + \alpha H(\pi(\cdot|\mathbf{s}_t)))\right]$$

其中 $\alpha$ 为温度参数，$H(\pi)$ 为策略分布的熵。

#### 2.3.2 Actor网络（Tanh-Squashed Gaussian策略）

Actor网络参数化高斯策略，输出经Tanh压缩的连续动作：

$$\mu_\phi(\mathbf{s}), \log\sigma_\phi(\mathbf{s}) = \text{Actor}_\phi(\mathbf{s})$$

采样过程（重参数化技巧）：
$$u \sim \mathcal{N}(\mu_\phi, \sigma_\phi^2), \quad \mathbf{a} = \tanh(u)$$

对数概率修正（Jacobian修正）：
$$\log \pi(\mathbf{a}|\mathbf{s}) = \log \pi(u|\mathbf{s}) - \sum_{i=1}^{d_a} \log(1 - a_i^2 + \epsilon)$$

其中 $d_a$ 为动作维度。

#### 2.3.3 Critic网络（双Q网络）

使用双Q网络缓解高估问题：

$$Q_{\theta_1}(\mathbf{s}, \mathbf{a}), Q_{\theta_2}(\mathbf{s}, \mathbf{a}) = \text{Critic}_\theta(\mathbf{s}, \mathbf{a})$$

目标Q值：
$$Q_{\text{target}} = r + \gamma(1-d)\min(Q_{\theta_1'}(\mathbf{s}', \mathbf{a}'), Q_{\theta_2'}(\mathbf{s}', \mathbf{a}')) - \alpha \log\pi(\mathbf{a}'|\mathbf{s}')$$

其中 $\mathbf{a}' \sim \pi_\phi(\cdot|\mathbf{s}')$，$d$ 为终止标志。

#### 2.3.4 自动熵调节

温度参数 $\alpha$ 通过最小化以下目标自动调节：

$$J_\alpha = -\mathbb{E}_{\mathbf{s} \sim \mathcal{D}}[\log \alpha (\log \pi(\mathbf{a}|\mathbf{s}) + H_0)]$$

其中 $H_0 = -d_a$ 为目标熵。

#### 2.3.5 训练更新规则

**Critic更新**：
$$\theta_i \leftarrow \theta_i - \lambda_Q \nabla_{\theta_i} \mathbb{E}_{(\mathbf{s},\mathbf{a},r,\mathbf{s}',d) \sim \mathcal{D}}[(Q_{\theta_i}(\mathbf{s}, \mathbf{a}) - Q_{\text{target}})^2]$$

**Actor更新**：
$$\phi \leftarrow \phi - \lambda_\pi \nabla_\phi \mathbb{E}_{\mathbf{s} \sim \mathcal{D}}[\alpha \log \pi_\phi(\mathbf{a}|\mathbf{s}) - Q_{\theta_1}(\mathbf{s}, \mathbf{a})]$$

**Alpha更新**：
$$\log \alpha \leftarrow \log \alpha - \lambda_\alpha \nabla_{\log \alpha} \mathbb{E}_{\mathbf{s} \sim \mathcal{D}}[-\log \alpha (\log \pi(\mathbf{a}|\mathbf{s}) + H_0)]$$

**软目标网络更新**：
$$\theta' \leftarrow \tau \theta + (1-\tau) \theta'$$

其中 $\tau = 0.005$ 为软更新系数。

### 2.4 宏命令执行策略 (Macro Command Execution)

为了降低决策频率并提高样本效率，采用宏命令执行策略：

**决策步** ($t \equiv 0 \pmod{K}$)：
1. 观测当前状态 $\mathbf{s}_t$
2. 从策略 $\pi_\phi$ 采样动作 $\mathbf{a}_t$
3. 缓存 $(\mathbf{s}_t, \mathbf{a}_t)$，重置累积奖励

**执行步** ($t \not\equiv 0 \pmod{K}$)：
1. 维持上一个宏命令 $\mathbf{a}_{t-K}$
2. 累积单步奖励：$R_{\text{accum}} \leftarrow R_{\text{accum}} + r_t$

**转移存储**：
$$(\mathbf{s}_{t-K}, \mathbf{a}_{t-K}, R_{\text{accum}}, \mathbf{s}_t) \rightarrow \text{ReplayBuffer}$$

其中 $K = 10$ 为动作间隔。

### 2.5 PSO粒子更新与小生境维护

#### 2.5.1 速度与位置更新

对于第 $g$ 个子种群的第 $i$ 个粒子：

$$\mathbf{v}_{g,i}^{(t+1)} = w_g \mathbf{v}_{g,i}^{(t)} + c_{1,g} r_1 (\mathbf{p}_{g,i} - \mathbf{x}_{g,i}^{(t)}) + c_{2,g} r_2 (\mathbf{g}^*_g - \mathbf{x}_{g,i}^{(t)})$$

$$\mathbf{x}_{g,i}^{(t+1)} = \mathbf{x}_{g,i}^{(t)} + \mathbf{v}_{g,i}^{(t+1)}$$

其中 $w_g, c_{1,g}, c_{2,g}$ 由SAC Agent动态调节。

#### 2.5.2 增强邻域差分变异 (Enhanced NDM)

当子种群陷入停滞时触发NDM算子：

$$\mathbf{x}_{g,i}^{\text{new}} = \mathbf{x}_{g,i} + F(\mathbf{g}_{\text{far}} - \mathbf{g}_{\text{near}}) + 0.1 \boldsymbol{\xi}$$

其中：
- $\mathbf{g}_{\text{far}} = \arg\max_i \|\mathbf{x}_{g,i} - \mathbf{g}^*_g\|_2$（最远粒子）
- $\mathbf{g}_{\text{near}} = \arg\min_i \|\mathbf{x}_{g,i} - \mathbf{g}^*_g\|_2$（最近粒子）
- $F \sim \text{Uniform}(0.5, 1.0)$（缩放因子）
- $\boldsymbol{\xi} \sim \mathcal{N}(0, 0.1^2 I)$（扰动）

应用于最差30%的粒子，以增强多样性。

### 2.6 密集奖励设计

奖励函数综合考虑多个目标：

$$r_g^{(t)} = w_1 r_{\text{fit}} + w_2 r_{\text{div}} + w_3 r_{\text{danger}} + w_4 r_{\text{niche}}$$

**适应度奖励**：
$$r_{\text{fit}} = \begin{cases}
1.0 & \text{if } f^*_g < f^*_{g,\text{prev}} \\
-0.1 & \text{otherwise}
\end{cases}$$

**多样性奖励**：
$$r_{\text{div}} = 0.5 \cdot \frac{D_{\text{curr}} - D_{\text{prev}}}{D_{\text{prev}} + \epsilon}$$

**危险发现奖励**：
$$r_{\text{danger}} = \begin{cases}
2.0 & \text{if } f^*_g < \tau_d \text{ and } f^*_{g,\text{prev}} \geq \tau_d \\
0.0 & \text{otherwise}
\end{cases}$$

**小生境覆盖奖励**：
$$r_{\text{niche}} = 0.3 \cdot \Delta \text{Coverage}$$

其中权重 $(w_1, w_2, w_3, w_4) = (0.4, 0.2, 0.3, 0.1)$。

### 2.7 算法流程

**算法 1：SAC驱动的状态感知小生境PSO**

```
输入：环境env，SAC Agent，粒子数N，子种群数G，最大迭代T
初始化：粒子位置、速度、子种群参数
for t = 1 to T do
    for g = 1 to G do
        s_g ← _get_state(g)  // 获取28维状态
        if t mod K == 0 then  // 决策步
            a_g ← π_φ(s_g)  // SAC采样动作
            缓存(s_g, a_g)，重置R_accum
        end if
        if Agent类型 == continuous then
            _apply_continuous_action(g, a_g)  // 应用连续参数
        else
            _apply_action(g, a_g)  // 应用离散动作
        end if
    end for
    
    PSO更新：速度、位置、适应度评估
    更新子种群最优
    
    if 停滞 then
        触发NDM算子
    end if
    
    计算奖励r_g
    R_accum ← R_accum + r_g
    
    if t mod K == 0 then  // 存储转移
        for g = 1 to G do
            ReplayBuffer.add((s_g, a_g, R_accum, s'_g))
        end for
    end if
    
    SAC更新：Critic、Actor、Alpha
    
end for
```

