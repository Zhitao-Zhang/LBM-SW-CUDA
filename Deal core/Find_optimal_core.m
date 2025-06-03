clc
clear all
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

folder1 = 'C:\core data\BaiduNetdiskDownload\TJ24G5027\1 sample 1000';
files1 = dir(fullfile(folder1, '*.tif'));
fileNames1 = {files1.name};
num_images = numel(fileNames1);
for k = 1:num_images
img = imread(fullfile(folder1, fileNames1{k}));
img0(:,:,k) = img;
end
divideSize = 100;

% Flatten the array to a 1D vector
img0_flat = img0(:);

% Filter out only non-zero elements
non_zero_elements = img0_flat(img0_flat ~= 0);

% Use histcounts to count occurrences of 1, 2, 3, 4, 5, 6
[counts, edges] = histcounts(non_zero_elements, 0.5:1:6.5); % Adjusted to include 5

% Calculate the total number of non-zero elements
total_non_zero = numel(non_zero_elements);

% Calculate the percentage of each element (1, 2, 3, 4, 5) in the non-zero set
percentages = (counts / total_non_zero) * 100;

% Display the counts and percentages relative to non-zero elements
disp(['Number of 1s: ', num2str(counts(1)), ' (', num2str(percentages(1)), '% of non-zero elements)']);
disp(['Number of 2s: ', num2str(counts(2)), ' (', num2str(percentages(2)), '% of non-zero elements)']);
disp(['Number of 3s: ', num2str(counts(3)), ' (', num2str(percentages(3)), '% of non-zero elements)']);
disp(['Number of 4s: ', num2str(counts(4)), ' (', num2str(percentages(4)), '% of non-zero elements)']);
disp(['Number of 5s: ', num2str(counts(5)), ' (', num2str(percentages(5)), '% of non-zero elements)']);
disp(['Number of 6s: ', num2str(counts(6)), ' (', num2str(percentages(6)), '% of non-zero elements)']);

%匹配最合适的体积占比
img1 = [];  % Initialize img1
best_diff = inf;  % Initialize best difference to a large value
best_position = [];  % To store the position of the best match
% Set step size for slicing (can be adjusted for performance)
step_size = 40;  % You can modify this for faster or more precise searching

% 定义目标非零占比
target_percentages = [19.559, 4.910, 22.014, 19.226, 26.711, 7.580];  % 对应元素 1 到 6

% 遍历 img0 的 x, y, z 维度以提取子数组
for x_start = 1:step_size:(size(img0, 1) - divideSize + 1)
    for y_start = 1:step_size:(size(img0, 2) - divideSize + 1)
        for z_start = 1:step_size:(size(img0, 3) - divideSize + 1)
            
            % 从 img0 中提取 400x400x400 的子数组
            img1_temp = img0(x_start:x_start+divideSize-1, ...
                             y_start:y_start+divideSize-1, ...
                             z_start:z_start+divideSize-1);
            
            % 将 img1_temp 展平成一维向量并过滤零元素
            img1_flat = img1_temp(:);
            
            % 检查 img1_temp 是否包含零元素
            if any(img1_flat == 0)
                continue;  % 如果包含零元素，跳过此次循环
            end
            
            % 计算非零元素中每个值的出现次数
            non_zero_elements1 = img1_flat;  % 由于已无零元素，直接使用
            [counts1, ~] = histcounts(non_zero_elements1, 0.5:1:6.5);
            
            % 计算非零元素的总数
            total_non_zero1 = numel(non_zero_elements1);
            
            % 计算每个元素（1 到 6）在非零元素中的百分比
            percentages1 = (counts1 / total_non_zero1) * 100;
            
            % 计算与目标百分比的差异
            diff = sum(abs(target_percentages - percentages1));
            
            % 如果此子数组的百分比更接近目标百分比，则保存
            if diff < best_diff
                best_diff = diff;
                best_position = [x_start, x_start+divideSize-1, ...
                                 y_start, y_start+divideSize-1, ...
                                 z_start, z_start+divideSize-1];
                img1 = img1_temp;  % 保存最佳匹配的子数组
                best_counts1 = counts1;  % 保存计数以供显示
                best_percentages1 = percentages1;  % 保存百分比以供显示
            end
        end
    end
end

% 显示最佳匹配的位置及对应的元素计数和百分比
disp(['最佳匹配位置：(', num2str(best_position(1)), ':', num2str(best_position(2)), ', ', ...
      num2str(best_position(3)), ':', num2str(best_position(4)), ', ', ...
      num2str(best_position(5)), ':', num2str(best_position(6)), ')']);
disp('最佳匹配的非零元素百分比：');
disp(['1 的数量：', num2str(best_counts1(1)), ' (', num2str(best_percentages1(1)), '%)']);
disp(['2 的数量：', num2str(best_counts1(2)), ' (', num2str(best_percentages1(2)), '%)']);
disp(['3 的数量：', num2str(best_counts1(3)), ' (', num2str(best_percentages1(3)), '%)']);
disp(['4 的数量：', num2str(best_counts1(4)), ' (', num2str(best_percentages1(4)), '%)']);
disp(['5 的数量：', num2str(best_counts1(5)), ' (', num2str(best_percentages1(5)), '%)']);
disp(['6 的数量：', num2str(best_counts1(6)), ' (', num2str(best_percentages1(6)), '%)']);

% Flatten the img1 array to a 1D vector
img0_flat1 = img1(:);

% Filter out only non-zero elements
non_zero_elements1 = img0_flat1(img0_flat1 ~= 0);

% Use histcounts to count occurrences of 1, 2, 3, 4, 5
[counts1, edges1] = histcounts(non_zero_elements1, 0.5:1:6.5); % Adjusted to include 5

% Calculate the total number of non-zero elements
total_non_zero1 = numel(non_zero_elements1);

% Calculate the percentage of each element (1, 2, 3, 4, 5) in the non-zero set
percentages1 = (counts1 / total_non_zero1) * 100;

% Display the counts and percentages relative to non-zero elements
disp(['Number of 1s in img1: ', num2str(counts1(1)), ' (', num2str(percentages1(1)), '% of non-zero elements)']);
disp(['Number of 2s in img1: ', num2str(counts1(2)), ' (', num2str(percentages1(2)), '% of non-zero elements)']);
disp(['Number of 3s in img1: ', num2str(counts1(3)), ' (', num2str(percentages1(3)), '% of non-zero elements)']);
disp(['Number of 4s in img1: ', num2str(counts1(4)), ' (', num2str(percentages1(4)), '% of non-zero elements)']);
disp(['Number of 5s in img1: ', num2str(counts1(5)), ' (', num2str(percentages1(5)), '% of non-zero elements)']);
disp(['Number of 6s in img1: ', num2str(counts1(6)), ' (', num2str(percentages1(6)), '% of non-zero elements)']);

% Count the number of zero elements in img1
zero_count11 = sum(img0_flat1 == 0);

% Display the count of zero elements in img1
disp(['Number of zero elements in img1: ', num2str(zero_count11)]);

%% 打开文件用于写入
outputFilename = '100x100x100.dat';
fileID = fopen(outputFilename, 'w');
% 检查文件是否成功打开
if fileID == -1
    error('Failed to open file %s for writing.', outputFilename);
end
% 预分配 a 的大小
numElements = size(img1, 1) * size(img1, 2) * size(img1, 3);
a = zeros(numElements, 4);

% 遍历切割后的图像的每个像素，存储坐标和值到 a 中
index = 1;
for i = 1:size(img1, 3)
    for x = 1:size(img1, 1)
        for y = 1:size(img1, 2)
            % 存储坐标和值
            a(index, :) = [i-1, y-1, x-1, img1(x, y, i)];
            % 写入坐标和值到文件
            fprintf(fileID, '%d %d %d %.6f\n', i-1, x-1, y-1, img1(x, y, i));
            index = index + 1;
        end
    end
    disp(i);
end
% 关闭文件
fclose(fileID);
