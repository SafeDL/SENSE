# SAC-PSO 关键数学公式补充

## 7. SAC算法的详细数学推导

### 7.1 最大熵强化学习目标

标准RL目标：
$$J(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{T} \gamma^t r(\mathbf{s}_t, \mathbf{a}_t)\right]$$

最大熵RL目标（加入策略熵正则化）：
$$J(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{T} \gamma^t (r(\mathbf{s}_t, \mathbf{a}_t) + \alpha H(\pi(\cdot|\mathbf{s}_t)))\right]$$

其中策略熵定义为：
$$H(\pi(\cdot|\mathbf{s})) = -\mathbb{E}_{\mathbf{a} \sim \pi(\cdot|\mathbf{s})}[\log \pi(\mathbf{a}|\mathbf{s})]$$

### 7.2 Soft Q-Learning

定义Soft Q函数：
$$Q^{\pi}(\mathbf{s}, \mathbf{a}) = \mathbb{E}_{(\mathbf{s}', \mathbf{a}') \sim \pi}\left[r(\mathbf{s}, \mathbf{a}) + \gamma(Q^{\pi}(\mathbf{s}', \mathbf{a}') + \alpha H(\pi(\cdot|\mathbf{s}')))\right]$$

Soft Bellman等式：
$$Q^{\pi}(\mathbf{s}, \mathbf{a}) = r(\mathbf{s}, \mathbf{a}) + \gamma \mathbb{E}_{\mathbf{s}' \sim p}[V^{\pi}(\mathbf{s}')] + \alpha H(\pi(\cdot|\mathbf{s}))$$

其中Soft V函数为：
$$V^{\pi}(\mathbf{s}) = \mathbb{E}_{\mathbf{a} \sim \pi(\cdot|\mathbf{s})}[Q^{\pi}(\mathbf{s}, \mathbf{a}) - \alpha \log \pi(\mathbf{a}|\mathbf{s})]$$

### 7.3 Actor-Critic更新

**Critic损失函数**（最小化Bellman残差）：
$$\mathcal{L}_Q(\theta) = \mathbb{E}_{(\mathbf{s},\mathbf{a},r,\mathbf{s}',d) \sim \mathcal{D}}\left[(Q_\theta(\mathbf{s}, \mathbf{a}) - (r + \gamma(1-d)\min_i Q_{\theta_i'}(\mathbf{s}', \mathbf{a}') - \alpha \log \pi_\phi(\mathbf{a}'|\mathbf{s}')))^2\right]$$

**Actor损失函数**（最大化Q值与熵）：
$$\mathcal{L}_\pi(\phi) = \mathbb{E}_{\mathbf{s} \sim \mathcal{D}, \mathbf{a} \sim \pi_\phi(\cdot|\mathbf{s})}\left[\alpha \log \pi_\phi(\mathbf{a}|\mathbf{s}) - Q_{\theta_1}(\mathbf{s}, \mathbf{a})\right]$$

**Alpha损失函数**（自动熵调节）：
$$\mathcal{L}_\alpha = \mathbb{E}_{\mathbf{s} \sim \mathcal{D}, \mathbf{a} \sim \pi_\phi(\cdot|\mathbf{s})}\left[-\log \alpha (\log \pi_\phi(\mathbf{a}|\mathbf{s}) + H_0)\right]$$

其中 $H_0 = -d_a$ 为目标熵。

### 7.4 Tanh-Squashed Gaussian策略

**无约束高斯分布**：
$$u \sim \mathcal{N}(\mu_\phi(\mathbf{s}), \sigma_\phi^2(\mathbf{s}))$$

**Tanh压缩**：
$$\mathbf{a} = \tanh(u) \in (-1, 1)^{d_a}$$

**对数概率修正**（使用变量替换）：

设 $u = \text{arctanh}(\mathbf{a})$，则：
$$\log \pi(\mathbf{a}|\mathbf{s}) = \log \pi(u|\mathbf{s}) - \sum_{i=1}^{d_a} \log(1 - a_i^2)$$

其中 $\pi(u|\mathbf{s}) = \mathcal{N}(u; \mu_\phi(\mathbf{s}), \sigma_\phi^2(\mathbf{s}))$

完整形式：
$$\log \pi(\mathbf{a}|\mathbf{s}) = -\frac{1}{2}\sum_{i=1}^{d_a}\left(\frac{(u_i - \mu_i)^2}{\sigma_i^2} + \log(2\pi\sigma_i^2)\right) - \sum_{i=1}^{d_a}\log(1 - a_i^2 + \epsilon)$$

## 8. PSO参数自适应机制

### 8.1 惯性权重 $w$ 的学习

**传统线性衰减**：
$$w_t = w_{\max} - \frac{(w_{\max} - w_{\min}) \cdot t}{T_{\max}}$$

**SAC学习的自适应**：
$$w_g^{(t+1)} = (1-\alpha) w_g^{(t)} + \alpha w_{\text{SAC}}$$

其中 $w_{\text{SAC}} \in [0.4, 0.9]$ 由Agent输出。

**学习效果指标**：
$$\Delta w = |w_{\text{SAC}} - w_{\text{default}}|$$

### 8.2 加速系数 $c_1, c_2$ 的学习

**认知系数** $c_1$（个体学习）：
- 高 $c_1$：强调个体最优，增加局部搜索
- 低 $c_1$：减弱个体影响，增加全局搜索

**社会系数** $c_2$（社会学习）：
- 高 $c_2$：强调群体最优，加速收敛
- 低 $c_2$：减弱群体影响，增加多样性

**平衡条件**：
$$c_1 + c_2 \approx 4.0 \quad \text{（Clerc-Kennedy条件）}$$

SAC学习的约束：
$$c_{1,g}^{(t+1)} + c_{2,g}^{(t+1)} \in [1.0, 4.0]$$

### 8.3 速度缩放因子 $v_{\text{scale}}$ 的学习

**基础最大速度**：
$$v_{\max} = 0.2 \times (\mathbf{x}_{\max} - \mathbf{x}_{\min})$$

**动态调整**：
$$v_{\max}^{(t)} = v_{\max} \times v_{\text{scale}}_g^{(t)}$$

其中 $v_{\text{scale}} \in [0.5, 2.0]$。

**物理意义**：
- $v_{\text{scale}} > 1$：增加探索范围
- $v_{\text{scale}} < 1$：减小探索范围，精细搜索

## 9. 多样性度量与奖励

### 9.1 位置多样性

**标准差多样性**：
$$D_{\text{pos}} = \frac{1}{d}\sum_{j=1}^{d} \text{std}(x_{:,j}^{\text{norm}})$$

其中 $x^{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$

**熵多样性**（离散化）：
$$D_{\text{entropy}} = -\sum_{k=1}^{m} p_k \log p_k$$

其中 $p_k$ 为第 $k$ 个网格单元的粒子占比。

### 9.2 速度多样性

**速度范数多样性**：
$$D_{\text{vel}} = \text{std}(\|\mathbf{v}_i\|_2)$$

**物理意义**：
- 高速度多样性：粒子运动方向差异大，探索能力强
- 低速度多样性：粒子运动方向一致，可能陷入局部最优

### 9.3 聚集度

**到质心距离**：
$$C = \frac{1}{n}\sum_{i=1}^{n} \|\mathbf{x}_i - \bar{\mathbf{x}}\|_2$$

**归一化聚集度**：
$$C_{\text{norm}} = \frac{C}{\sqrt{d}}$$

其中 $\sqrt{d}$ 为搜索空间对角线长度。

## 10. 小生境覆盖率计算

### 10.1 网格划分

将搜索空间 $\mathcal{X} = [-1, 1]^d$ 均匀划分为 $m$ 个网格单元：

$$\mathcal{G} = \{g_1, g_2, \ldots, g_m\}$$

对于3维空间，使用 $\sqrt[3]{m} = 5$ 的网格分辨率，得到 $m = 125$ 个单元。

但在实际应用中，基于危险点分布统计，有效单元数为 $m_{\text{eff}} = 97$。

### 10.2 覆盖率定义

**原始覆盖率**：
$$\text{Cov}_{\text{raw}} = \frac{|\{g_i : \exists \mathbf{x} \in g_i, f(\mathbf{x}) < \tau_d\}|}{m}$$

**有效覆盖率**（基于真实危险单元）：
$$\text{Cov}_{\text{eff}} = \frac{|\{g_i : \exists \mathbf{x} \in g_i, f(\mathbf{x}) < \tau_d\}|}{m_{\text{eff}}}$$

### 10.3 增量覆盖率

**单步增量**：
$$\Delta \text{Cov}^{(t)} = \text{Cov}^{(t)} - \text{Cov}^{(t-1)}$$

**累积增量**：
$$\text{Cov}_{\text{total}} = \sum_{t=1}^{T} \Delta \text{Cov}^{(t)}$$

## 11. 性能指标的数学定义

### 11.1 收敛速度

**对数收敛率**：
$$r_{\text{conv}} = \frac{\log(f_0 - f_{\text{opt}}) - \log(f_T - f_{\text{opt}})}{T}$$

其中 $f_0$ 为初始最优值，$f_T$ 为第 $T$ 步最优值。

### 11.2 探索-利用平衡

**探索指数**：
$$E = \frac{D_{\text{curr}}}{D_{\text{init}}}$$

其中 $D_{\text{init}}$ 为初始多样性。

- $E \approx 1$：保持初始多样性，强探索
- $E \ll 1$：多样性快速下降，强利用

### 11.3 稳定性指标

**方差稳定性**：
$$\text{Stability} = 1 - \frac{\text{std}(\text{Cov}_{\text{runs}})}{\text{mean}(\text{Cov}_{\text{runs}})}$$

其中 $\text{Cov}_{\text{runs}}$ 为多次独立运行的覆盖率。

值域 $[0, 1]$，越接近1表示稳定性越好。

