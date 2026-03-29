%% Multi-Method Training Reward Comparison (Raw + MA, no right axis)
% Style: light (raw) + dark (MA) for each method
% Data source: ../rlsan/src/RLSearch/results/<ALGO>_*/<matfile>

clear; clc; close all;

%% ------------------------ Config ------------------------
cfg = struct();
cfg.data_root  = fullfile('..','rlsan','src','RLSearch','results');

% 你可以在这里“增加更多方法”
% 规则：会去 results 里找 "<KEY>_*" 文件夹，然后加载里面的 mat 文件
% mat 默认用 lower(key) + "_training_data.mat"
cfg.algos = struct( ...
    'key',   {'SAC','PPO','DDPG','TD3','TRPO'}, ...
    'label', {'SAC','PPO','DDPG','TD3','TRPO'}, ...
    'mat',   {'sac_training_data.mat','ppo_training_data.mat','ddpg_training_data.mat','td3_training_data.mat','trpo_training_data.mat'} ...
);

% 若你不想手动写 mat 文件名，可自动生成：
% e.g. SAC -> sac_training_data.mat
for i = 1:numel(cfg.algos)
    if ~isfield(cfg.algos(i),'mat') || isempty(cfg.algos(i).mat)
        cfg.algos(i).mat = [lower(cfg.algos(i).key) '_training_data.mat'];
    end
end

cfg.fontName = 'Times New Roman';
cfg.fontSize = 12;

cfg.ma_window = 10;     % 平滑窗口（保持与你 SAC 脚本一致）
cfg.reward_ylim = [];   % 为空则自动；若要固定如 [0 0.6]，直接赋值

% “像你附图”的视觉参数
cfg.raw_lighten_alpha = 0.65;  % 越大越浅（0~1）
cfg.raw_lw = 0.6;
cfg.ma_lw  = 2.2;

cfg.save_fig = true;
cfg.out_name = 'fig_reward_multi_methods';

%% ------------------------ Load data (like RL-PSO script) ------------------------
loaded = loadAllAlgos(cfg);

fprintf('\nTotal loaded: %d methods\n', loaded.n);
if loaded.n == 0
    error('No training data found. Check cfg.data_root and folder naming: <ALGO>_*/<matfile>.');
end

%% ------------------------ Plot: Reward (Raw + MA) ------------------------
fig = figure('Color','w','Position',[200, 200, 900, 520]);
ax  = axes(fig); hold(ax,'on'); box(ax,'on'); grid(ax,'on');

h_leg = gobjects(loaded.n,1);

for i = 1:loaded.n
    [x, y] = getXY(loaded.data{i}, 'episodes', 'mean_rewards');
    if isempty(x), continue; end

    % Moving average (保持核心逻辑不变)
    y_ma = movmean(y, cfg.ma_window, 'Endpoints','shrink');

    c  = loaded.colors(i,:);
    cL = lightenColor(c, cfg.raw_lighten_alpha);

    % Raw (浅色细线，不进 legend)
    plot(ax, x, y, ...
        'Color', cL, ...
        'LineWidth', cfg.raw_lw, ...
        'LineStyle','-', ...
        'HandleVisibility','off');

    % MA (深色粗线，实线，进 legend)
    h_leg(i) = plot(ax, x, y_ma, ...
        'Color', c, ...
        'LineWidth', cfg.ma_lw, ...
        'LineStyle','-', ...
        'DisplayName', sprintf('%s  MA(%d)', loaded.labels{i}, cfg.ma_window));
end

% Y 轴范围：自动加 margin（更稳），或使用 cfg.reward_ylim 固定
if isempty(cfg.reward_ylim)
    allY = [];
    for i = 1:loaded.n
        if isfield(loaded.data{i},'mean_rewards')
            allY = [allY, double(loaded.data{i}.mean_rewards(:))']; %#ok<AGROW>
        end
    end
    y_min = min(allY); y_max = max(allY);
    y_margin = (y_max - y_min) * 0.08;
    if y_margin < 1e-6, y_margin = 0.01; end
    ylim(ax, [y_min - y_margin, y_max + y_margin]);
else
    ylim(ax, cfg.reward_ylim);
end

% Axes style（保持你 SAC 版本的“实线网格”）
set(ax, 'FontName', cfg.fontName, 'FontSize', cfg.fontSize, ...
    'GridLineStyle','-', 'GridAlpha',0.30, ...
    'TickDir','out', 'LineWidth',0.9);

xlabel(ax, 'Episode', 'FontSize', cfg.fontSize+1, ...
    'FontName', cfg.fontName, 'FontWeight','bold');
ylabel(ax, 'Mean Reward', 'FontSize', cfg.fontSize+1, ...
    'FontName', cfg.fontName, 'FontWeight','bold');

title(ax, 'Training Reward Comparison (Raw + Moving Average)', ...
    'FontSize', cfg.fontSize+3, 'FontName', cfg.fontName, 'FontWeight','bold');

leg_h = h_leg(h_leg ~= 0);
if ~isempty(leg_h)
    lgd = legend(ax, leg_h, 'Location','southeast', ...
        'FontName', cfg.fontName, 'FontSize', cfg.fontSize-1, 'Box','on');
    lgd.EdgeColor = [0.7 0.7 0.7];
end

%% ------------------------ Save ------------------------
if cfg.save_fig
    exportgraphics(fig, [cfg.out_name '.png'], 'Resolution', 300);
    fprintf('Saved: %s.png\n', cfg.out_name);

    exportgraphics(fig, [cfg.out_name '.pdf'], 'ContentType','vector');
    fprintf('Saved: %s.pdf\n', cfg.out_name);
end

fprintf('Done.\n');

%% ========================= Local Functions =========================
function loaded = loadAllAlgos(cfg)
    if ~isfolder(cfg.data_root)
        error('data_root folder not found: %s', cfg.data_root);
    end

    d = dir(cfg.data_root);
    d = d([d.isdir]);
    d = d(~ismember({d.name},{'.','..'}));

    labels  = {};
    data    = {};
    colors  = [];
    n = 0;

    % 自动给颜色：方法多了也不怕
    baseColors = lines(max(7, numel(cfg.algos))); % 保证颜色够用

    for i = 1:numel(cfg.algos)
        key = cfg.algos(i).key;

        match_idx = find(startsWith({d.name}, [key '_']), 1, 'first');
        if isempty(match_idx)
            fprintf('[SKIP] %s: folder "<%s_*>” not found in %s\n', key, key, cfg.data_root);
            continue;
        end

        folder_name = d(match_idx).name;
        mat_path = fullfile(cfg.data_root, folder_name, cfg.algos(i).mat);

        if ~exist(mat_path,'file')
            fprintf('[SKIP] %s: mat missing in %s -> %s\n', key, folder_name, cfg.algos(i).mat);
            continue;
        end

        S = load(mat_path);
        if ~isfield(S,'episodes') || ~isfield(S,'mean_rewards')
            fprintf('[SKIP] %s: missing fields episodes/mean_rewards -> %s\n', key, mat_path);
            continue;
        end

        n = n + 1;
        labels{n} = cfg.algos(i).label; %#ok<AGROW>
        data{n}   = S;                 %#ok<AGROW>
        colors(n,:) = baseColors(mod(n-1,size(baseColors,1))+1,:); %#ok<AGROW>

        fprintf('[LOADED] %-6s <- %s/%s\n', cfg.algos(i).label, folder_name, cfg.algos(i).mat);
    end

    loaded = struct('n',n,'labels',{labels},'data',{data},'colors',colors);
end

function [x, y] = getXY(S, xfield, yfield)
    x = []; y = [];
    if ~isfield(S,xfield) || ~isfield(S,yfield), return; end
    x = double(S.(xfield)(:))';
    y = double(S.(yfield)(:))';
    m = min(numel(x), numel(y));
    x = x(1:m); y = y(1:m);
end

function c = lightenColor(rgb, alpha)
    alpha = max(0, min(1, alpha));
    c = rgb + (1 - rgb) * alpha;
end