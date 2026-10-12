clc
clear all

%%========================================================================
% 1. 读取整套图像（img0）
%%========================================================================
folder1 = 'C:\core data\BaiduNetdiskDownload\TJ24G5027\1 sample 1000';

files1 = dir(fullfile(folder1, '*.tif'));
fileNames1 = {files1.name};
num_images = numel(fileNames1);

for k = 1:num_images
    img = imread(fullfile(folder1, fileNames1{k}));
    img0(:, :, k) = img;  %#ok<SAGROW>
end

%%========================================================================
% 2. 设置随机切割参数
%%========================================================================

divideSize = 900;      % 子块尺寸：divideSize × divideSize × divideSize
n = 5;                 % 随机截取 5 个子块

% 是否要求子块内部全部非零
require_nonzero = true;

% 为了每次运行都产生不同的随机结果
rng('shuffle');

%%========================================================================
% 3. 显示整个模型尺寸及矿物比例
%%========================================================================

sz = size(img0);

fprintf('========================================\n');
fprintf('原始模型尺寸：%d × %d × %d\n', sz(1), sz(2), sz(3));
fprintf('随机子块尺寸：%d × %d × %d\n', ...
        divideSize, divideSize, divideSize);
fprintf('随机截取数量：%d\n', n);
fprintf('========================================\n');

% 检查 divideSize 是否超过模型尺寸
if any(sz < divideSize)
    error('divideSize 大于原始模型某一维度，无法进行切割。');
end

% 原始模型矿物统计
img0_flat = img0(:);
non_zero_elements = img0_flat(img0_flat ~= 0);

[counts_all, ~] = histcounts(non_zero_elements, 0.5:1:6.5);

total_non_zero = numel(non_zero_elements);

if total_non_zero > 0

    percentages_all = counts_all / total_non_zero * 100;

    disp('=== 整体 img0 中各矿物（1~6）的数量及占非零元素百分比 ===');

    for i = 1:6
        fprintf('元素 %d: %d (%.2f%%)\n', ...
                i, counts_all(i), percentages_all(i));
    end

end

%%========================================================================
% 4. 随机选择 5 个子块
%%========================================================================

random_positions = zeros(n, 6);
random_subvolumes = cell(1, n);

count_valid = 0;
count_try = 0;

% 防止极端情况下无限循环
max_try = 100000;

while count_valid < n

    count_try = count_try + 1;

    if count_try > max_try
        error(['随机尝试超过 %d 次，仍然没有找到足够的有效子块。\n' ...
               '可能原因：divideSize 太大或者模型中 0 元素过多。'], ...
               max_try);
    end

    %%--------------------------------------------------------------------
    % 随机生成起始位置
    %%--------------------------------------------------------------------

    x_start = randi([1, sz(1) - divideSize + 1]);
    y_start = randi([1, sz(2) - divideSize + 1]);
    z_start = randi([1, sz(3) - divideSize + 1]);

    x_end = x_start + divideSize - 1;
    y_end = y_start + divideSize - 1;
    z_end = z_start + divideSize - 1;

    %%--------------------------------------------------------------------
    % 检查是否与已经选中的起始位置完全相同
    %%--------------------------------------------------------------------

    if count_valid > 0

        old_starts = random_positions(1:count_valid, [1 3 5]);

        current_start = [x_start, y_start, z_start];

        if any(all(old_starts == current_start, 2))
            continue;
        end

    end

    %%--------------------------------------------------------------------
    % 提取子块
    %%--------------------------------------------------------------------

    img_sub = img0( ...
        x_start:x_end, ...
        y_start:y_end, ...
        z_start:z_end);

    %%--------------------------------------------------------------------
    % 如果要求全部非零，则检查是否存在 0
    %%--------------------------------------------------------------------

    if require_nonzero

        if any(img_sub(:) == 0)
            continue;
        end

    end

    %%--------------------------------------------------------------------
    % 接受该随机子块
    %%--------------------------------------------------------------------

    count_valid = count_valid + 1;

    random_positions(count_valid, :) = ...
        [x_start, x_end, ...
         y_start, y_end, ...
         z_start, z_end];

    random_subvolumes{count_valid} = img_sub;

    fprintf(['找到随机子块 No.%d：' ...
             'x[%d:%d], y[%d:%d], z[%d:%d]\n'], ...
             count_valid, ...
             x_start, x_end, ...
             y_start, y_end, ...
             z_start, z_end);

end

fprintf('\n总随机尝试次数：%d\n', count_try);

%%========================================================================
% 5. 输出随机子块信息
%%========================================================================

disp(' ');
disp('=== 随机选出的 5 个子块 ===');

for i = 1:n

    pos = random_positions(i, :);

    fprintf(['No.%d: ' ...
             'x[%d:%d], y[%d:%d], z[%d:%d]\n'], ...
             i, ...
             pos(1), pos(2), ...
             pos(3), pos(4), ...
             pos(5), pos(6));

end

%%========================================================================
% 6. 将 5 个随机子块写入 dat 文件
%%========================================================================

for i = 1:n

    img1 = random_subvolumes{i};

    %%--------------------------------------------------------------------
    % 统计该子块中的矿物比例
    %%--------------------------------------------------------------------

    counts_i = histcounts(img1(:), 0.5:1:6.5);

    total_i = numel(img1);

    pct_i = counts_i / total_i * 100;

    fprintf('\n========================================\n');
    fprintf('随机子块 No.%d 统计\n', i);
    fprintf('========================================\n');

    for j = 1:6

        fprintf('元素 %d: 数量 = %d，占比 = %.3f%%\n', ...
                j, counts_i(j), pct_i(j));

    end

    %%--------------------------------------------------------------------
    % 输出文件
    %%--------------------------------------------------------------------

    outputFilename = sprintf( ...
        'model_random_%d_%dx%dx%d.dat', ...
        i, ...
        divideSize, divideSize, divideSize);

    fileID = fopen(outputFilename, 'w');

    if fileID == -1
        error('无法打开文件 %s 进行写入。', outputFilename);
    end

    %%--------------------------------------------------------------------
    % 写入格式：
    %
    % z  x  y  value
    %
    % 坐标从 0 开始
    %%--------------------------------------------------------------------

    [sx, sy, sz2] = size(img1);

    for k2 = 1:sz2

        for x2 = 1:sx

            for y2 = 1:sy

                value = img1(x2, y2, k2);

                fprintf(fileID, ...
                        '%d %d %d %d\n', ...
                        k2 - 1, ...
                        x2 - 1, ...
                        y2 - 1, ...
                        value);

            end

        end

    end

    fclose(fileID);

    fprintf('已写出：%s\n', outputFilename);

end

disp(' ');
disp('=== 5 个随机子块已全部写出完毕 ===');
