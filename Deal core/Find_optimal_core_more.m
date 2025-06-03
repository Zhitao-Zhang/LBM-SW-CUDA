clc
clear all

%%========================================================================
% 1. 定义全局参数和目标百分比
%%========================================================================
com1 = 1.75e12;
com2 = 4.54e11;
com3 = 1.85e12;
com4 = 1.84e12;
com5 = 2.29e12;
com6 = 6.01e11;
all = com1 + com2 + com3 + com4 + com5 + com6;
per_com1 = com1 / all;
per_com2 = com2 / all;
per_com3 = com3 / all;
per_com4 = com4 / all;
per_com5 = com5 / all;
per_com6 = com6 / all;

% 目标百分比（1~6 对应的百分比）
target_percentages = [19.559, 4.910, 22.014, 19.226, 26.711, 7.580];

%%========================================================================
% 2. 读取整套图像（img0），并将非零元素比例打印出来（可选）
%%========================================================================
folder1 = 'C:\core data\BaiduNetdiskDownload\TJ24G5027\1 sample 1000';
files1 = dir(fullfile(folder1, '*.tif'));
fileNames1 = {files1.name};
num_images = numel(fileNames1);

for k = 1:num_images
    img = imread(fullfile(folder1, fileNames1{k}));
    img0(:, :, k) = img;  %#ok<SAGROW>
end

divideSize = 100;       % 要切出的子块尺寸：300×300×300
step_size = 40;         % 采样步长（可调整）

% ——（可选）先统计原始整块 img0 中非零元素的分布
img0_flat = img0(:);
non_zero_elements = img0_flat(img0_flat ~= 0);
[counts_all, ~] = histcounts(non_zero_elements, 0.5:1:6.5);
total_non_zero = numel(non_zero_elements);
percentages_all = (counts_all / total_non_zero) * 100;
disp('=== 整体 img0 中各矿物（1~6）的数量及占非零元素百分比 ===');
for i = 1:6
    fprintf('Number of %d''s: %d (%.2f%%)\n', i, counts_all(i), percentages_all(i));
end

%%========================================================================
% 3. 初始化“前 n 个候选”列表
%%========================================================================
n = 5;  % 想要取前 n 个最匹配的子块

% best_diffs 保存目前前 n 个子块的 diff 值，初始为 inf
best_diffs = inf(1, n);

% best_positions 保存每个候选子块的坐标： 
% 我们用一个 n×6 的矩阵，每一行记录 [x_start, x_end, y_start, y_end, z_start, z_end]
best_positions = zeros(n, 6);

% 为了最终写文件，还需要一个 cell 数组保存每个子块本身，以便后续输出
best_subvolumes = cell(1, n);

%%========================================================================
% 4. 在三重循环中枚举所有可能的子块，计算 diff，并维护“前 n 最优”列表
%%========================================================================
sz = size(img0);
countLoop = 0;
for x_start = 1:step_size:(sz(1) - divideSize + 1)
    for y_start = 1:step_size:(sz(2) - divideSize + 1)
        for z_start = 1:step_size:(sz(3) - divideSize + 1)
            countLoop = countLoop + 1;
            % 提取子块
            x_end = x_start + divideSize - 1;
            y_end = y_start + divideSize - 1;
            z_end = z_start + divideSize - 1;
            img_sub = img0(x_start:x_end, y_start:y_end, z_start:z_end);

            % 将子块展开并检查零元素
            flat_sub = img_sub(:);
            if any(flat_sub == 0)
                continue;  % 只考虑全非零的子块
            end

            % 统计子块中各矿物（1~6）的分布
            [cnt_sub, ~] = histcounts(flat_sub, 0.5:1:6.5);
            total_sub = numel(flat_sub);
            pct_sub = (cnt_sub / total_sub) * 100;

            % 计算与目标百分比的差异之和
            diff_val = sum(abs(target_percentages - pct_sub));

            % 如果比当前候选列表里最差（max(best_diffs)）更好，则替换并排序
            [worst_diff, worst_idx] = max(best_diffs);
            if diff_val < worst_diff
                % 把新候选放到最差的位置
                best_diffs(worst_idx) = diff_val;
                best_positions(worst_idx, :) = [x_start, x_end, y_start, y_end, z_start, z_end];
                best_subvolumes{worst_idx} = img_sub;  % 暂存子块

                % 对 best_diffs 及对应的位置信息按升序排序
                [best_diffs, sort_idx] = sort(best_diffs, 'ascend');
                best_positions = best_positions(sort_idx, :);
                best_subvolumes = best_subvolumes(sort_idx);
            end
        end
    end
end
disp(['=== 枚举结束，共检索到 ', num2str(countLoop), ' 个候选位置（按步长 ', num2str(step_size), '） ===']);

% 显示最终挑选出的前 n 个子块的 diff 值和位置信息
disp('=== 前 n 个最优子块信息（按 diff 从小到大排列） ===');
for i = 1:n
    pos = best_positions(i, :);
    fprintf('No.%d: diff = %.4f; 位置: x[%d:%d], y[%d:%d], z[%d:%d]\n', ...
            i, best_diffs(i), pos(1), pos(2), pos(3), pos(4), pos(5), pos(6));
end

%%========================================================================
% 5. 将这 n 个子块逐个写入文件：model_1_300x300x300.dat ... model_n_300x300x300.dat
%%========================================================================
for i = 1:n
    img1 = best_subvolumes{i};
    pos = best_positions(i, :);
    counts_i = histcounts(img1(:), 0.5:1:6.5);
    pct_i = (counts_i / numel(img1(:))) * 100;

    % 输出该子块的计数与百分比（可选）
    fprintf('\n--- 子块 No.%d (diff=%.4f) 统计 ---\n', i, best_diffs(i));
    for j = 1:6
        fprintf('  元素 %d: 数量 = %d (%.2f%%)\n', j, counts_i(j), pct_i(j));
    end

    % 创建输出文件
    outputFilename = sprintf('model_%d_%dx%dx%d.dat', i, divideSize, divideSize, divideSize);
    fileID = fopen(outputFilename, 'w');
    if fileID == -1
        error('无法打开文件 %s 进行写入。', outputFilename);
    end

    % 将子块的 (z, x, y, value) 按行写入
    idx = 1;
    [sx, sy, sz2] = size(img1);  % 都应为 divideSize
    for k2 = 1:sz2                     % z 维度，对应 i-1
        for x2 = 1:sx                  % x 维度，对应 y-1
            for y2 = 1:sy              % y 维度，对应 x-1
                value = img1(x2, y2, k2);
                % 写入格式：z (从 0 开始), x (从 0 开始), y (从 0 开始), value
                fprintf(fileID, '%d %d %d %d\n', k2-1, x2-1, y2-1, value);
                idx = idx + 1;
            end
        end
    end

    fclose(fileID);
    fprintf('已将子块 No.%d 写入文件：%s\n', i, outputFilename);
end

disp('=== 所有前 n 个子块已全部写出完毕 ===');
