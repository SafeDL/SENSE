%% RL-PSO Training Curves Comparison (Optimized)
% Plot reward / fitness / action params for PPO, SAC, DDPG, TD3, TRPO

clear; clc; close all;

%% ------------------------- Config -------------------------
cfg = struct();

% Root folder: change to your actual path
cfg.data_root = fullfile('..', 'rlsan', 'src', 'RLSearch', 'results');

cfg.algos = struct( ...
    'key',   {'PPO','SAC','DDPG','TD3','TRPO'}, ...
    'label', {'PPO','SAC','DDPG','TD3','TRPO'}, ...
    'mat',   {'ppo_training_data.mat','sac_training_data.mat','ddpg_training_data.mat','td3_training_data.mat','trpo_training_data.mat'} ...
);

cfg.colors = [ ...
    0.0000, 0.4470, 0.7410; ...
    0.8500, 0.3250, 0.0980; ...
    0.4660, 0.6740, 0.1880; ...
    0.9290, 0.6940, 0.1250; ...
    0.4940, 0.1840, 0.5560  ...
];
cfg.markers = {'o','s','d','^','v'};

cfg.fontName     = 'Times New Roman';
cfg.marker_step  = 50;
cfg.smooth_win   = 10;

cfg.reward_ylim  = [0, 0.6];
cfg.inset_max_ep = 100;

cfg.tail_n_box   = 50;   % last N episodes for boxplot
cfg.last_n_bar   = 20;   % last N episodes for bar mean/std

%% ------------------------- Load data -------------------------
loaded = loadAllAlgos(cfg);

fprintf('Total loaded: %d algorithms\n\n', loaded.n);
if loaded.n == 0
    error('No training data found. Check cfg.data_root and folder naming: <ALGO>_*/<matfile>.');
end

%% ------------------------- Figure 1: Average Return (Raw + Smoothed) -------------------------
fig1 = figure('Color','w','Position',[100, 100, 900, 550]);
main_ax = axes('Position',[0.10, 0.12, 0.85, 0.82]);
hold(main_ax,'on'); box(main_ax,'on'); grid(main_ax,'on');

% 你可以调这两个参数来更像附图
raw_lighten = 0.80;     % Raw 曲线“变浅”程度（越大越浅，0~1）
raw_lw      = 0.8;      % Raw 线宽
ma_lw       = 2.0;      % 平滑线宽

h_leg = gobjects(loaded.n,1);

for i = 1:loaded.n
    [x, y] = getXY(loaded.data{i}, 'episodes', 'mean_rewards');
    if isempty(x), continue; end

    % 平滑（移动平均），并保留原始数据
    y_ma = movmean(y, cfg.smooth_win);

    c  = loaded.colors(i,:);
    cL = lightenColor(c, raw_lighten);

    % Raw（浅色、细线，不进 legend）
    plot(main_ax, x, y, ...
        'Color', cL, ...
        'LineWidth', raw_lw, ...
        'HandleVisibility','off');

    % MA（深色、粗线，进 legend）
    h_leg(i) = plot(main_ax, x, y_ma, ...
        'Color', c, ...
        'LineWidth', ma_lw, ...
        'DisplayName', sprintf('%s  MA(%d)', loaded.labels{i}, cfg.smooth_win));

    % （可选）若你仍想保留 marker 点，可对 MA 画 marker（更像你之前的风格）
    mk = cfg.marker_step : cfg.marker_step : numel(x);
    if ~isempty(mk)
        plot(main_ax, x(mk), y_ma(mk), loaded.markers{i}, ...
            'Color', c, 'MarkerFaceColor', c, ...
            'MarkerSize', 5, 'HandleVisibility','off');
    end
end

styleAxes(main_ax, cfg);
xlabel(main_ax,'Episode','FontSize',13,'FontWeight','bold');
ylabel(main_ax,'Mean Reward','FontSize',13,'FontWeight','bold');
ylim(main_ax, cfg.reward_ylim);
title(main_ax,'Training Reward (Raw + Moving Average)','FontSize',14,'FontWeight','bold');

leg_h = h_leg(h_leg ~= 0);
if ~isempty(leg_h)
    legend(leg_h, 'Location','southeast', 'FontSize',10, 'FontName',cfg.fontName, ...
        'Box','on', 'EdgeColor',[0.7 0.7 0.7]);
end

saveas(fig1,'fig_reward_comparison.png');
fprintf('Figure 1 saved.\n');

%% ------------------------- Figure 2: Best Fitness Convergence -------------------------
fig2 = figure('Color','w','Position',[150, 150, 900, 550]);
ax2 = axes(fig2); hold(ax2,'on'); box(ax2,'on'); grid(ax2,'on');

h2 = gobjects(loaded.n,1);
for i = 1:loaded.n
    if ~isfield(loaded.data{i}, 'best_fitness'), continue; end
    [x, bf] = getXY(loaded.data{i}, 'episodes', 'best_fitness');
    cum_best = cummin(bf);

    % light raw curve (avoid 4-channel RGBA for older MATLAB versions)
    plot(ax2, x, bf, 'Color', lightenColor(loaded.colors(i,:), 0.85), ...
        'LineWidth', 0.5, 'HandleVisibility','off');

    h2(i) = plot(ax2, x, cum_best, 'Color', loaded.colors(i,:), ...
        'LineWidth', 2.0, 'DisplayName', loaded.labels{i});
end

styleAxes(ax2, cfg);
xlabel(ax2,'Training Episodes','FontSize',13,'FontWeight','bold');
ylabel(ax2,'Best Fitness (Cumulative Min)','FontSize',13,'FontWeight','bold');
title(ax2,'RL-PSO Training: Best Fitness Convergence','FontSize',14,'FontWeight','bold');

yline(ax2, -0.3, '--', 'Danger Threshold', 'Color',[0.8 0.2 0.2], ...
    'LineWidth',1.2, 'FontSize',9, 'HandleVisibility','off');

leg_h2 = h2(h2 ~= 0);
if ~isempty(leg_h2)
    legend(leg_h2, 'Location','northeast', 'FontSize',10, 'FontName',cfg.fontName, 'Box','on');
end

saveas(fig2,'fig_fitness_comparison.png');
fprintf('Figure 2 saved.\n');

%% ------------------------- Figure 3: Action Parameter Evolution (2x2) -------------------------
param_fields = {'action_w','action_c1','action_c2','action_vs'};
param_titles = {'Inertia Weight (w)','Cognitive Coef. (c1)','Social Coef. (c2)','Velocity Scale (vs)'};
param_ylabs  = {'w','c1','c2','vs'};

fig3 = figure('Color','w','Position',[200, 200, 1000, 700]);

for p = 1:4
    ax = subplot(2,2,p); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
    h3 = gobjects(loaded.n,1);

    for i = 1:loaded.n
        if ~isfield(loaded.data{i}, param_fields{p}), continue; end
        [x, y] = getXY(loaded.data{i}, 'episodes', param_fields{p});
        y_s = movmean(y, cfg.smooth_win);
        h3(i) = plot(ax, x, y_s, 'Color', loaded.colors(i,:), ...
            'LineWidth', 1.8, 'DisplayName', loaded.labels{i});
    end

    styleAxes(ax, cfg, 10);
    xlabel(ax,'Training Episodes','FontSize',12);
    ylabel(ax, param_ylabs{p}, 'FontSize',12);
    title(ax, param_titles{p}, 'FontSize',12);

    if p == 4
        leg_h3 = h3(h3 ~= 0);
        if ~isempty(leg_h3)
            legend(leg_h3, 'Location','best', 'FontSize',9, 'FontName',cfg.fontName, 'Box','on');
        end
    end
end

sgtitle('RL-PSO: Action Parameter Evolution','FontSize',14,'FontWeight','bold','FontName',cfg.fontName);
saveas(fig3,'fig_action_params.png');
fprintf('Figure 3 saved.\n');

%% ------------------------- Figure 4: Convergence Summary (boxplot + bar) -------------------------
fig4 = figure('Color','w','Position',[250, 250, 900, 500]);

% Left: boxplot of last N episodes
ax4a = subplot(1,2,1); hold(ax4a,'on'); box(ax4a,'on'); grid(ax4a,'on');

box_data = [];
box_groups = [];

for i = 1:loaded.n
    if ~isfield(loaded.data{i}, 'mean_rewards'), continue; end
    y = double(loaded.data{i}.mean_rewards(:))';
    t_start = max(1, numel(y) - cfg.tail_n_box + 1);
    tail = y(t_start:end);
    box_data = [box_data, tail]; %#ok<AGROW>
    box_groups = [box_groups, i * ones(1, numel(tail))]; %#ok<AGROW>
end

if ~isempty(box_data)
    bp = boxplot(ax4a, box_data, box_groups, ...
        'Labels', loaded.labels, 'Widths', 0.5, 'Symbol','+');
    set(bp, 'LineWidth', 1.5);
end

styleAxes(ax4a, cfg);
ylabel(ax4a,'Average Return','FontSize',13,'FontWeight','bold');
title(ax4a, sprintf('Reward Distribution (Last %d Ep)', cfg.tail_n_box), ...
    'FontSize',13,'FontWeight','bold');

% Right: bar chart final performance
ax4b = subplot(1,2,2); hold(ax4b,'on'); box(ax4b,'on'); grid(ax4b,'on');

bar_m = nan(loaded.n,1);
bar_s = nan(loaded.n,1);

for i = 1:loaded.n
    if ~isfield(loaded.data{i}, 'mean_rewards'), continue; end
    y = double(loaded.data{i}.mean_rewards(:))';
    ln = min(cfg.last_n_bar, numel(y));
    seg = y(end-ln+1:end);
    bar_m(i) = mean(seg);
    bar_s(i) = std(seg);
end

bh = bar(ax4b, 1:loaded.n, bar_m, 0.6);
bh.FaceColor = 'flat';
for i = 1:loaded.n
    bh.CData(i,:) = loaded.colors(i,:);
end
errorbar(ax4b, 1:loaded.n, bar_m, bar_s, 'k.', 'LineWidth',1.2, 'CapSize',8);

for i = 1:loaded.n
    if isnan(bar_m(i)), continue; end
    text(ax4b, i, bar_m(i)+bar_s(i)+0.002, sprintf('%.3f', bar_m(i)), ...
        'HorizontalAlignment','center', 'FontSize',9, 'FontName',cfg.fontName, 'FontWeight','bold');
end

set(ax4b,'XTick',1:loaded.n,'XTickLabel',loaded.labels);
styleAxes(ax4b, cfg);
ylabel(ax4b,'Average Return','FontSize',13,'FontWeight','bold');
title(ax4b, sprintf('Final Performance (Last %d Ep)', cfg.last_n_bar), ...
    'FontSize',13,'FontWeight','bold');

saveas(fig4,'fig_convergence_summary.png');
fprintf('Figure 4 saved.\n');

%% ------------------------- Console Summary -------------------------
fprintf('\n========================================\n');
fprintf('  RL-PSO Performance Summary\n');
fprintf('========================================\n');
fprintf('  Algo  | Ep  | Final R | Std R  | Best\n');
fprintf('--------+-----+---------+--------+------\n');

for i = 1:loaded.n
    if ~isfield(loaded.data{i}, 'mean_rewards')
        fprintf('  %-5s |  -- |   --    |  --    |  --\n', loaded.labels{i});
        continue;
    end

    y = double(loaded.data{i}.mean_rewards(:))';
    nep = numel(y);

    ln = min(cfg.last_n_bar, nep);
    seg = y(end-ln+1:end);
    mN = mean(seg);
    sN = std(seg);

    bestc = NaN;
    if isfield(loaded.data{i}, 'best_fitness')
        bestc = min(double(loaded.data{i}.best_fitness(:)));
    end

    fprintf('  %-5s | %3d | %+.4f | %.4f | %+.4f\n', loaded.labels{i}, nep, mN, sN, bestc);
end

fprintf('========================================\n');
fprintf('All figures generated.\n');

%% ========================= Local Functions =========================
function loaded = loadAllAlgos(cfg)
    % Validate root
    if ~isfolder(cfg.data_root)
        error('data_root folder not found: %s', cfg.data_root);
    end

    d = dir(cfg.data_root);
    d = d([d.isdir]);
    d = d(~ismember({d.name},{'.','..'}));

    labels = {};
    data   = {};
    colors = [];
    markers = {};
    n = 0;

    for i = 1:numel(cfg.algos)
        key = cfg.algos(i).key;

        % find folder starting with "<KEY>_"
        match_idx = find(startsWith({d.name}, [key '_']), 1, 'first');
        if isempty(match_idx)
            continue;
        end

        folder_name = d(match_idx).name;
        mat_path = fullfile(cfg.data_root, folder_name, cfg.algos(i).mat);

        if ~exist(mat_path,'file')
            fprintf('[SKIP] %s folder found (%s) but mat missing: %s\n', key, folder_name, cfg.algos(i).mat);
            continue;
        end

        S = load(mat_path);
        % Basic field sanity (episodes + mean_rewards are used most)
        if ~isfield(S,'episodes') || ~isfield(S,'mean_rewards')
            fprintf('[SKIP] %s mat exists but missing required fields episodes/mean_rewards: %s\n', key, mat_path);
            continue;
        end

        n = n + 1;
        labels{n}  = cfg.algos(i).label; %#ok<AGROW>
        data{n}    = S;                 %#ok<AGROW>
        colors(n,:) = cfg.colors(i,:);  %#ok<AGROW>
        markers{n} = cfg.markers{i};    %#ok<AGROW>

        fprintf('[LOADED] %s <- %s\n', cfg.algos(i).label, folder_name);
    end

    loaded = struct('n',n,'labels',{labels},'data',{data},'colors',colors,'markers',{markers});
end

function [x, y] = getXY(S, xfield, yfield)
    % Extract, cast, and align lengths safely
    x = [];
    y = [];
    if ~isfield(S, xfield) || ~isfield(S, yfield), return; end

    x = double(S.(xfield)(:))';
    y = double(S.(yfield)(:))';

    m = min(numel(x), numel(y));
    x = x(1:m);
    y = y(1:m);
end

function styleAxes(ax, cfg, fontSize)
    if nargin < 3, fontSize = 11; end
    set(ax, 'FontSize', fontSize, 'FontName', cfg.fontName, 'LineWidth', 1.0);
    set(ax, 'GridLineStyle', '--', 'GridAlpha', 0.3);
end

function c = lightenColor(rgb, alpha)
    % alpha in [0,1], larger = lighter (closer to white)
    alpha = max(0,min(1,alpha));
    c = rgb + (1-rgb)*alpha;
end