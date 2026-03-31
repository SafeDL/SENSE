%% 代理模型性能对比图 (顶刊学术风格)
% 绘制不同机器学习方法的F1-Score曲线对比 (横坐标: 训练迭代次数)
clear; clc; close all;

%% 1. 加载数据
% 加载其他ML方法的结果
% 假设 iterations 和 rf_f1, mlp_f1, xgb_f1, knn_f1, random_f1 在.mat文件中
% 这里使用模拟数据，确保代码可运行。请在使用时将此处注释掉，并加载实际数据。
% 模拟数据（使用实际数据时请删除以下模拟部分）
iterations = 1:500;

% --- 实际使用请取消注释并确保文件存在 ---
load('surrogate_comparison.mat');
% 加载GP方法的结果
if exist('gp_surrogate_result.mat', 'file')
    gp_data = load('gp_surrogate_result.mat');
    has_gp = true;
else
    has_gp = false;
    warning('gp_surrogate_result.mat not found. Run build_surrogate.py first.');
end
% -----------------------------------------

%% 2. 集中管理模型信息 (顶刊风格配色和线型)
% 定义学术风格配色方案
colors.gp = [0.84, 0.19, 0.15];       % 深红/珊瑚色 (gp (ours))
colors.rf = [0.27, 0.46, 0.70];       % 中蓝
colors.xgb = [0.10, 0.60, 0.31];      % 深绿
colors.mlp = [0.99, 0.55, 0.35];      % 橙色
colors.knn = [0.57, 0.75, 0.86];      % 浅蓝
colors.random = [0.000, 0.000, 0.000]; % 中灰色 (基准线)

% 定义学术风格线型
linestyles.gp = '-'; linestyles.rf = '-'; linestyles.xgb = '--';
linestyles.mlp = '-.'; linestyles.knn = ':'; linestyles.random = '-';

% 定义学术风格线宽
linewidths.gp = 1.5; linewidths.rf = 1.5; linewidths.xgb = 1.5;
linewidths.mlp = 1.5; linewidths.knn = 1.5; linewidths.random = 1.0;

% 集中管理结构体，按图例顺序排列 (GP在最前以突出)
model_info = struct();
model_info.gp_data = struct('name', 'gp (ours)', 'data', gp_data.gp_f1, 'color', colors.gp, 'ls', linestyles.gp, 'lw', linewidths.gp);
model_info.rf = struct('name', 'rf', 'data', rf_f1, 'color', colors.rf, 'ls', linestyles.rf, 'lw', linewidths.rf);
model_info.mlp = struct('name', 'mlp', 'data', mlp_f1, 'color', colors.mlp, 'ls', linestyles.mlp, 'lw', linewidths.mlp);
model_info.xgb = struct('name', 'xgb', 'data', xgb_f1, 'color', colors.xgb, 'ls', linestyles.xgb, 'lw', linewidths.xgb);
model_info.knn = struct('name', 'knn', 'data', knn_f1, 'color', colors.knn, 'ls', linestyles.knn, 'lw', linewidths.knn);
model_info.random = struct('name', 'random', 'data', random_f1, 'color', colors.random, 'ls', linestyles.random, 'lw', linewidths.random);

% 定义绘图顺序（随机方法在底层）
draw_order = {'random', 'knn', 'xgb', 'mlp', 'rf', 'gp_data'};

%% 3. 绘图和图形格式化
% 创建更专业的图形窗口，隐藏界面元素，设置白色背景
fig = figure('Position', [100, 100, 600, 500],'Color', 'white');
hold on;

h_plot = zeros(1, length(draw_order));
% 绘图循环
for i = 1:length(draw_order)
    m = draw_order{i};
    h_plot(i) = plot(iterations, model_info.(m).data, ...
        'Color', model_info.(m).color, ...
        'LineStyle', model_info.(m).ls, ...
        'LineWidth', model_info.(m).lw);
end

% 添加标题
title('Performance Comparison of Surrogate Models', 'FontSize', 16, 'FontWeight', 'bold', 'FontName', 'Arial');

%% 4. 图例和坐标轴增强
% 创建清晰、专业的图例，顺序为学术惯例 (GP在前)，突出显示
legend_order = {'gp_data', 'rf', 'mlp', 'xgb', 'knn', 'random'};
legend_handles = zeros(1, length(legend_order));
legend_names = cell(1, length(legend_order));
for i = 1:length(legend_order)
    legend_handles(i) = h_plot(strcmp(draw_order, legend_order{i}));
    legend_names{i} = model_info.(legend_order{i}).name;
end

leg = legend(legend_handles, legend_names, 'Location', 'southeast', 'FontSize', 12, 'FontName', 'Arial', 'Box', 'on', 'EdgeColor', [0.8 0.8 0.8], 'Color', [1 1 1], 'NumColumns', 2);

% 坐标轴标签增强
xlabel('Iteration', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Arial');
ylabel('F1-Score', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Arial');

% 设置坐标轴范围和格式
xlim([0, 500]);
ylim([0, 1.0]);
set(gca, 'Box', 'on', 'LineWidth', 1.0, 'FontName', 'Arial', 'FontSize', 11, 'TickDir', 'in');

% 网格增强
grid on;
set(gca, 'GridLineStyle', '-', 'GridAlpha', 0.15, 'MinorGridLineStyle', 'none');

% 保存图片（可选，顶刊出版请使用300dpi以上）
% exportgraphics(fig, '代理模型精度对比_顶刊版.jpg', 'Resolution', 300);