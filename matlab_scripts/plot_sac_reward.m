%% Plot SAC Training Reward (Raw + MA)
% Style: light-blue Raw curve + dark-blue MA curve (solid line)
% Data: sac_training_data.mat

clear; clc; close all;

%% ------------------------ Config ------------------------
mat_file  = 'sac_training_data.mat';   % .mat file path
ma_window = 10;                        % moving average window
save_fig  = true;                      % save PNG/PDF
out_name  = 'fig_sac_reward';          % output name (no extension)

font_name = 'Times New Roman';
font_size = 12;

% Colors (RGB)
color_raw = [0.55, 0.63, 0.83];   % light blue (raw)
color_ma  = [0.20, 0.25, 0.60];   % dark blue (MA)

% Visual tuning
raw_lighten_alpha = 0.65;   % 0~1, larger => closer to white (simulate transparency)
raw_lw = 0.6;
ma_lw  = 2.2;

%% ------------------------ Load data ------------------------
if ~isfile(mat_file)
    error('数据文件未找到: %s', mat_file);
end

S = load(mat_file);

% Required fields check (only reward-related)
req_fields = {'episodes','mean_rewards'};
for k = 1:numel(req_fields)
    if ~isfield(S, req_fields{k})
        error('MAT 文件缺少字段: %s', req_fields{k});
    end
end

episodes     = double(S.episodes(:))';
mean_rewards = double(S.mean_rewards(:))';

% Align lengths robustly
N = min(numel(episodes), numel(mean_rewards));
episodes     = episodes(1:N);
mean_rewards = mean_rewards(1:N);

fprintf('Loaded %d episodes from %s\n', N, mat_file);

%% ------------------------ Moving average ------------------------
ma_rewards = movmean(mean_rewards, ma_window, 'Endpoints','shrink');

%% ------------------------ Plot ------------------------
fig = figure('Color','w','Position',[200, 200, 720, 450]);
ax  = axes(fig); hold(ax,'on'); box(ax,'on'); grid(ax,'on');

% Raw curve: lighten color to mimic transparency (avoid RGBA incompatibility)
raw_color_vis = lightenColor(color_raw, raw_lighten_alpha);
h_raw = plot(ax, episodes, mean_rewards, ...
    'Color', raw_color_vis, ...
    'LineWidth', raw_lw, ...
    'LineStyle', '-');   % raw is solid

% MA curve: MUST be solid line (not dashed)
h_ma = plot(ax, episodes, ma_rewards, ...
    'Color', color_ma, ...
    'LineWidth', ma_lw, ...
    'LineStyle', '-');   % MA is solid

% Reward y-limits with margin
y_min = min(mean_rewards);
y_max = max(mean_rewards);
y_margin = (y_max - y_min) * 0.08;
if y_margin < 1e-6, y_margin = 0.01; end
ylim(ax, [y_min - y_margin, y_max + y_margin]);

% Labels
ylabel(ax, 'Mean Reward', 'FontSize', font_size+1, ...
    'FontName', font_name, 'FontWeight','bold');
xlabel(ax, 'Episode', 'FontSize', font_size+1, ...
    'FontName', font_name, 'FontWeight','bold');

% Axes styling (grid solid)
set(ax, 'FontName', font_name, 'FontSize', font_size, ...
    'GridLineStyle','-', 'GridAlpha',0.30, ...
    'TickDir','out', 'LineWidth',0.9);

title(ax, 'Training Reward', 'FontSize', font_size+3, ...
    'FontName', font_name, 'FontWeight','bold');

% Legend
lgd = legend(ax, [h_raw, h_ma], {'Raw', sprintf('MA(%d)', ma_window)}, ...
    'Location','southeast', 'FontName',font_name, 'FontSize',font_size-1, ...
    'Box','on');
lgd.EdgeColor = [0.7 0.7 0.7];

%% ------------------------ Save ------------------------
if save_fig
    exportgraphics(fig, [out_name '.png'], 'Resolution', 300);
    fprintf('Saved: %s.png\n', out_name);

    exportgraphics(fig, [out_name '.pdf'], 'ContentType','vector');
    fprintf('Saved: %s.pdf\n', out_name);
end

fprintf('Done.\n');

%% ------------------------ Local function ------------------------
function c = lightenColor(rgb, alpha)
% alpha in [0,1], larger -> closer to white
alpha = max(0, min(1, alpha));
c = rgb + (1 - rgb) * alpha;
end