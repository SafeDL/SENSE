%% 代理模型性能对比图
% 绘制不同机器学习方法的F1-Score曲线对比 (横坐标: 训练迭代次数)

clear; clc; close all;

%% 加载数据
% 加载其他ML方法的结果
load('surrogate_comparison.mat');

% 加载GP方法的结果 (需要单独运行build_surrogate.py生成)
if exist('gp_surrogate_result.mat', 'file')
    gp_data = load('gp_surrogate_result.mat');
    has_gp = true;
else
    has_gp = false;
    warning('gp_surrogate_result.mat not found. Run build_surrogate.py first.');
end

%% 颜色定义
colors = struct();
colors.knn = [0.0, 0.5, 0.8];       % 蓝色虚线
colors.mlp = [0.8, 0.6, 0.0];       % 橙色虚线 (Random Forest)
colors.random = [0.8, 0.5, 0.8];    % 粉色实线
colors.rf = [0.3, 0.3, 0.8];       % 深蓝色实线
colors.xgb = [0.2, 0.6, 0.2];       % 深绿色实线 
colors.gp = [0.8, 0.2, 0.2];       % 红色实线(本文方法)

%% 绘图
figure('Position', [100, 100, 600, 450]);
hold on;

% 虚线方法 (传统代理模型)
h1 = plot(iterations, rf_f1, '-', 'Color', colors.rf, 'LineWidth', 1.5);
h2 = plot(iterations, mlp_f1, '-', 'Color', colors.mlp, 'LineWidth', 1.5);

% 实线方法 (机器学习方法)
h3 = plot(iterations, xgb_f1, '-', 'Color', colors.xgb, 'LineWidth', 1.5);
h4 = plot(iterations, knn_f1, '-', 'Color', colors.knn, 'LineWidth', 1.5);
h5 = plot(iterations, random_f1, '-', 'Color', colors.random, 'LineWidth', 1.5);

% 本文GP方法 (粗实线突出显示)

h6 = plot(gp_data.iterations, gp_data.gp_f1, '-', 'Color', colors.gp, 'LineWidth', 1.5);
legend([h1, h2, h3, h4, h5, h6], ...
    {'rf', 'mlp', 'xgb', 'knn', 'random', 'gp (ours)'}, ...
    'Location', 'southeast', 'FontSize', 10);

%% 图例和标签
xlabel('Iteration', 'FontSize', 12);
ylabel('F1-Score', 'FontSize', 12);

% 设置坐标轴范围
xlim([0, max(iterations)+1]);
ylim([0, 1.0]);

% 网格
grid on;
set(gca, 'GridLineStyle', '-', 'GridAlpha', 0.3);
% 保存图片
% export_fig 代理模型精度对比.jpg -r 300
