%% SAC Multi-Run Training Visualization
% 绘制多次独立实验的均值 ± 标准差阴影图 (RL论文标准做法)
%
% 数据来源: sac_multi_run_data.mat (由 train_sac.py --num_runs 生成)
% 输出:
%   子图1 (左): 累计奖励曲线 + 阴影方差区域
%   子图2 (右): PSO 动作参数调节图 (w, c1, c2, vel_scale)

clear; clc; close all;

%% ======================== Config ========================
mat_file  = 'sac_multi_run_data.mat';
ma_window = 10;        % 移动平均窗口
out_name  = 'fig_sac_multi_run';

font_name = 'Times New Roman';
font_size = 12;

% 奖励曲线配色
color_mean  = [0.20, 0.25, 0.60];   % 深蓝 (均值线)
color_shade = [0.55, 0.63, 0.83];   % 浅蓝 (阴影区域)
shade_alpha = 0.30;                  % 阴影透明度

% 动作参数配色 (w, c1, c2, vel_scale)
action_colors = [
    0.12, 0.47, 0.71;   % 蓝 (w)
    0.89, 0.10, 0.11;   % 红 (c1)
    0.17, 0.63, 0.17;   % 绿 (c2)
    1.00, 0.50, 0.00;   % 橙 (vel_scale)
];
action_shade_alpha = 0.20;

%% ======================== Load Data ========================
if ~isfile(mat_file)
    error('数据文件未找到: %s\n请先运行 train_sac.py --num_runs N', mat_file);
end

S = load(mat_file);

% 必要字段检查
req_fields = {'episodes', 'all_rewards', 'all_action_w', ...
              'all_action_c1', 'all_action_c2', 'all_action_vs'};
for k = 1:numel(req_fields)
    if ~isfield(S, req_fields{k})
        error('MAT 文件缺少字段: %s', req_fields{k});
    end
end

episodes    = double(S.episodes(:))';
all_rewards = double(S.all_rewards);       % (num_runs x num_episodes)
all_w       = double(S.all_action_w);
all_c1      = double(S.all_action_c1);
all_c2      = double(S.all_action_c2);
all_vs      = double(S.all_action_vs);

[num_runs, num_eps] = size(all_rewards);
fprintf('已加载 %d 次独立实验, 每次 %d episodes\n', num_runs, num_eps);

%% ======================== 对每个 run 做移动平均 ========================
all_rewards_ma = zeros(size(all_rewards));
all_w_ma       = zeros(size(all_w));
all_c1_ma      = zeros(size(all_c1));
all_c2_ma      = zeros(size(all_c2));
all_vs_ma      = zeros(size(all_vs));

for r = 1:num_runs
    all_rewards_ma(r,:) = movmean(all_rewards(r,:), ma_window, 'Endpoints','shrink');
    all_w_ma(r,:)       = movmean(all_w(r,:),       ma_window, 'Endpoints','shrink');
    all_c1_ma(r,:)      = movmean(all_c1(r,:),      ma_window, 'Endpoints','shrink');
    all_c2_ma(r,:)      = movmean(all_c2(r,:),      ma_window, 'Endpoints','shrink');
    all_vs_ma(r,:)      = movmean(all_vs(r,:),       ma_window, 'Endpoints','shrink');
end

%% ======================== 计算 cross-run 统计量 ========================
reward_mean = mean(all_rewards_ma, 1);
reward_std  = std(all_rewards_ma, 0, 1);

w_mean  = mean(all_w_ma, 1);   w_std  = std(all_w_ma, 0, 1);
c1_mean = mean(all_c1_ma, 1);  c1_std = std(all_c1_ma, 0, 1);
c2_mean = mean(all_c2_ma, 1);  c2_std = std(all_c2_ma, 0, 1);
vs_mean = mean(all_vs_ma, 1);  vs_std = std(all_vs_ma, 0, 1);

%% ======================== Figure 1: SAC 奖励曲线 ========================
fig1 = figure(1);
set(fig1, 'Color', 'w', 'Position', [100, 200, 680, 500], 'Name', 'SAC Training Reward');

ax1 = axes(fig1);
hold(ax1, 'on'); box(ax1, 'on'); grid(ax1, 'on');

% 阴影区域 (mean ± 1 std)
upper = reward_mean + reward_std;
lower = reward_mean - reward_std;
fill_x = [episodes, fliplr(episodes)];
fill_y = [upper, fliplr(lower)];
h_shade = fill(ax1, fill_x, fill_y, color_shade, ...
    'FaceAlpha', shade_alpha, 'EdgeColor', 'none');


% 均值线
h_mean = plot(ax1, episodes, reward_mean, ...
    'Color', color_mean, 'LineWidth', 2.2, 'LineStyle', '-');


% 样式
set(ax1, 'FontName', font_name, 'FontSize', font_size, ...
    'LineWidth', 1.0, 'GridLineStyle', '-', 'GridAlpha', 0.25, ...
    'TickDir', 'out');
xlabel(ax1, 'Episode', 'FontSize', font_size+1, ...
    'FontName', font_name, 'FontWeight', 'bold');
ylabel(ax1, 'Mean Reward', 'FontSize', font_size+1, ...
    'FontName', font_name, 'FontWeight', 'bold');
title(ax1, sprintf('SAC Training Reward (%d runs)', num_runs), ...
    'FontSize', font_size+2, 'FontName', font_name, 'FontWeight', 'bold');

lgd1 = legend(ax1, [h_mean, h_shade], ...
    {sprintf('Mean (MA%d)', ma_window), '± 1 Std'}, ...
    'Location', 'southeast', 'FontName', font_name, ...
    'FontSize', font_size-1, 'Box', 'on');
lgd1.EdgeColor = [0.7 0.7 0.7];

%% ======================== Figure 2: PSO 动作参数 ========================
fig2 = figure(2);
set(fig2, 'Color', 'w', 'Position', [820, 200, 680, 500], 'Name', 'PSO Action Parameters');

ax2 = axes(fig2);
hold(ax2, 'on'); box(ax2, 'on'); grid(ax2, 'on');

param_means = {w_mean, c1_mean, c2_mean, vs_mean};
param_stds  = {w_std,  c1_std,  c2_std,  vs_std};
param_names = {'w (Inertia)', 'c_1 (Cognitive)', 'c_2 (Social)', 'v_s (Vel Scale)'};

h_params = gobjects(4, 1);

for p = 1:4
    mu = param_means{p};
    c  = action_colors(p, :);
    
    % 仅绘制均值线，不绘制误差带
    h_params(p) = plot(ax2, episodes, mu, ...
        'Color', c, 'LineWidth', 1.8, 'LineStyle', '-', ...
        'DisplayName', param_names{p});
end

% 样式
set(ax2, 'FontName', font_name, 'FontSize', font_size, ...
    'LineWidth', 1.0, 'GridLineStyle', '-', 'GridAlpha', 0.25, ...
    'TickDir', 'out');
xlabel(ax2, 'Episode', 'FontSize', font_size+1, ...
    'FontName', font_name, 'FontWeight', 'bold');
ylabel(ax2, 'Parameter Value', 'FontSize', font_size+1, ...
    'FontName', font_name, 'FontWeight', 'bold');
title(ax2, sprintf('PSO Action Parameters (%d runs)', num_runs), ...
    'FontSize', font_size+2, 'FontName', font_name, 'FontWeight', 'bold');

lgd2 = legend(ax2, h_params, 'Location', 'best', ...
    'FontName', font_name, 'FontSize', font_size-1, 'Box', 'on', ...
    'NumColumns', 2);
lgd2.EdgeColor = [0.7 0.7 0.7];

%% ======================== Console Summary ========================
fprintf('\n========================================\n');
fprintf('  SAC Multi-Run Summary\n');
fprintf('========================================\n');
fprintf('  Runs: %d | Episodes: %d | MA Window: %d\n', num_runs, num_eps, ma_window);
fprintf('  Final Reward: %.4f ± %.4f\n', reward_mean(end), reward_std(end));
fprintf('  Final w:      %.4f ± %.4f\n', w_mean(end), w_std(end));
fprintf('  Final c1:     %.4f ± %.4f\n', c1_mean(end), c1_std(end));
fprintf('  Final c2:     %.4f ± %.4f\n', c2_mean(end), c2_std(end));
fprintf('  Final vs:     %.4f ± %.4f\n', vs_mean(end), vs_std(end));
fprintf('========================================\n');
fprintf('Done.\n');
