"""
加载测试得分,分析其得分分布状态
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import yeojohnson, norm, probplot, shapiro, gamma, lognorm, genpareto, expon
from statsmodels.genmod.families import Tweedie
from statsmodels.api import GLM, add_constant

# 加载得分数据
def load_scores(file_path):
    with open(file_path, 'rb') as f:
        scores = pickle.load(f)
    return np.array(scores)

# 绘制直方图并拟合多种长尾分布
def plot_histogram_and_fit_distributions(scores, bins=50):
    # 绘制原始数据直方图
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=bins, density=True, alpha=0.6, color='g', label='Histogram')

    # 定义 x 轴范围
    x = np.linspace(scores.min(), scores.max(), 1000)

    # 初始化结果存储
    results = {}

    # 1. 拟合伽马分布
    gamma_params = gamma.fit(scores)
    gamma_pdf = gamma.pdf(x, *gamma_params)
    results['Gamma'] = gamma.logpdf(scores, *gamma_params).sum()
    plt.plot(x, gamma_pdf, 'r-', lw=2, label='Gamma Fit')

    # 2. 拟合对数正态分布
    lognorm_params = lognorm.fit(scores, floc=0)  # floc=0 确保分布从 0 开始
    lognorm_pdf = lognorm.pdf(x, *lognorm_params)
    results['LogNormal'] = lognorm.logpdf(scores, *lognorm_params).sum()
    plt.plot(x, lognorm_pdf, 'b-', lw=2, label='LogNormal Fit')

    # 3. 拟合广义帕累托分布
    genpareto_params = genpareto.fit(scores)
    genpareto_pdf = genpareto.pdf(x, *genpareto_params)
    results['GenPareto'] = genpareto.logpdf(scores, *genpareto_params).sum()
    plt.plot(x, genpareto_pdf, 'm-', lw=2, label='GenPareto Fit')

    # 4. 拟合指数分布
    expon_params = expon.fit(scores)
    expon_pdf = expon.pdf(x, *expon_params)
    results['Exponential'] = expon.logpdf(scores, *expon_params).sum()
    plt.plot(x, expon_pdf, 'c-', lw=2, label='Exponential Fit')

    # 5. 使用 Tweedie 分布拟合数据
    X = np.ones_like(scores)  # GLM 的自变量（常数项）
    y = scores  # GLM 的因变量
    tweedie_family = Tweedie(var_power=1.5)  # 假设 var_power=1.5（复合泊松-伽马分布）
    model = GLM(y, X, family=tweedie_family).fit()
    tweedie_mu = model.mu[0]  # 拟合的均值
    tweedie_phi = model.scale  # 拟合的尺度参数
    tweedie_pdf = (x ** (tweedie_family.var_power - 1)) * np.exp(-x / tweedie_mu) / tweedie_phi
    results['Tweedie'] = model.llf  # 对数似然值
    plt.plot(x, tweedie_pdf, 'y-', lw=2, label='Tweedie Fit (var_power=1.5)')

    # 设置图形标题和标签
    plt.title('Score Distribution and Multiple Fits', fontsize=16)
    plt.xlabel('Score', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.show()

    # 打印拟合结果
    print("\nLog-Likelihoods for Different Distributions:")
    for dist, log_likelihood in results.items():
        print(f"{dist}: {log_likelihood}")

    # 返回最佳分布
    best_fit = max(results, key=results.get)
    print(f"\nBest Fit Distribution: {best_fit}")
    return best_fit

# Yeo-Johnson变换并拟合正态分布
def yeo_johnson_and_normal_fit(scores):
    # 对得分数据进行 Yeo-Johnson 变换
    scores_transformed, lmbda = yeojohnson(scores)
    print(f"Yeo-Johnson Lambda: {lmbda}")

    # 绘制变换后数据的直方图
    plt.figure(figsize=(10, 6))
    plt.hist(scores_transformed, bins=50, density=True, alpha=0.6, color='b', label='Transformed Histogram')

    # 拟合正态分布
    mu, sigma = norm.fit(scores_transformed)
    print(f"Fitted Normal Distribution: mu={mu}, sigma={sigma}")

    # 生成正态分布曲线
    x = np.linspace(scores_transformed.min(), scores_transformed.max(), 1000)
    pdf_normal = norm.pdf(x, mu, sigma)

    # 绘制正态分布拟合曲线
    plt.plot(x, pdf_normal, 'r-', lw=2, label='Normal Fit')

    # 设置图形标题和标签
    plt.title('Yeo-Johnson Transformed Data and Normal Fit', fontsize=16)
    plt.xlabel('Transformed Score', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.show()

    # 正态性检验
    print("\nPerforming Normality Test:")
    stat, p_value = shapiro(scores_transformed)
    print(f"Shapiro-Wilk Test: Statistic={stat}, p-value={p_value}")

    if p_value > 0.05:
        print("The transformed data follows a normal distribution (p > 0.05).")
    else:
        print("The transformed data does not follow a normal distribution (p <= 0.05).")

    # Q-Q 图检验正态性
    plt.figure(figsize=(8, 6))
    probplot(scores_transformed, dist="norm", plot=plt)
    plt.title('Q-Q Plot for Yeo-Johnson Transformed Data', fontsize=16)
    plt.grid(alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # 替换为你的得分数据文件路径
    file_path = '../../data/scenario02_scores.pkl'
    scores = load_scores(file_path)

    # 绘制直方图并拟合多种分布
    best_fit = plot_histogram_and_fit_distributions(scores)

    # 对数据进行 Yeo-Johnson 变换并拟合正态分布
    yeo_johnson_and_normal_fit(scores)

