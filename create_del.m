close all;
clear;
clc
warning off;

addpath(genpath('ClusteringMeasure'));
addpath(genpath('utils'));
addpath(genpath('./para_result/'));
savepath;
which ClusteringMeasure8;

folder_path = "../../datasets/";
data_name = "MSRC_v1"; % 获取文件名
data_addr = folder_path+data_name;

load(data_addr);

if strcmp(data_name, 'MSRC_v1')==1
    load(data_addr);
    X = fea;
    Y = gt;
else
    [X,Y] = dataset_load(data_addr, data_name);
end
numview = length(X);

data_iter = 20;
for i = 1:numview
    X{i} = NormalizeFea(X{i},0);
    X{i}=X{i}';
end
NC=length(unique(Y));%number of category
num_sample=length(Y);

%-------------------------parameter-----------------------------%
anchor = 27;
beta = 1;
gamma = 0.1;
lambda = 1;
per_list = [0.1, 0.3, 0.5];
per = per_list(1);

max_ACC = 0;
for j = 1:data_iter
    [ind_folds_temp, ~]=get_incomplete(num_sample,per,numview);
    for i = 1:5
        [Zor,~] = GenerateZ_2(X,anchor,ind_folds_temp);
        [U,H, Obj, J_matrix] = EIMCAGC(Zor,NC,beta,gamma,lambda,ind_folds_temp);
        [~ , label] = max(H, [], 2);
        temp_result(i,:) = round(ClusteringMeasure(Y, label),4);
    end
    result(1,:) = [anchor, beta, gamma, lambda, round(mean(temp_result),4)];
    fprintf("ACC = %.4f", result(1,5));
    if(result(1,5)>max_ACC)
        max_ACC = result(1,5);
        ind_folds = ind_folds_temp;
    end
end

fprintf("----------------------最优的结果为：--------------------------------\n")
fprintf("ACC = %.4f", max_ACC);

file_path = '../incomplete_datasets/';
file_name = data_name + "_Del=" + num2str(per) + ".mat";
% 使用save函数将数据保存到指定路径
save(file_path+file_name, 'X', 'Y', 'ind_folds'); 

