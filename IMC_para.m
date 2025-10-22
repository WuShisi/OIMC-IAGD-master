close all;
clear;
clc
warning off;

addpath(genpath('ClusteringMeasure'));
addpath(genpath('utils'));
addpath(genpath('./para_new/'));

folder_path = './incomplete_datasets/';
files_list = dir(fullfile(folder_path,'*.mat'));
per_list = [0.1, 0.3, 0.5];

num_files = numel(files_list); 
for f = 1:num_files
    
    data_name = files_list(f).name; 
    data_addr = fullfile(files_list(f).folder, data_name); 
    
    load(data_addr);
    numview = length(X);
    for i = 1:numview
        X{i} = NormalizeFea(X{i},0);
    end 
    NC=length(unique(Y));
    num_sample=length(Y);
   
    %-------------------------parameter-----------------------------%
    anchor_list=[NC+5, NC+10, NC+20, NC+50]; %numebr of anchor 
    beta_list=[1e-2,1e-1,1e0,1e1,1e2];
    gamma_list=[1e-2,1e-1,1e0,1e1,1e2];
    lambda_list=[1e-2,1e-1,1e0,1e1,1e2];

    row = 1;
    for anchor_iter = 1:length(anchor_list)
        anchor = anchor_list(anchor_iter);
        for beta_iter = 1:length(beta_list)
            beta = beta_list(beta_iter);
            for gamma_iter = 1:length(gamma_list)
                gamma = gamma_list(gamma_iter);
                for lambda_iter = 1:length(lambda_list)
                    for i = 1:2
                        lambda = lambda_list(lambda_iter);
                        tic
                        [Zor] = GenerateZ_2(X,anchor,ind_folds);
                        [U,H, Obj] = EIMCAGC(Zor,NC,beta,gamma,lambda,ind_folds);
                        time(i) = toc;
                        [~ , label] = max(H, [], 2);
                        temp_result(i,:) = round(ClusteringMeasure8(Y, label),4);
                    end
                    result(row,:) = [anchor, beta, gamma, lambda, round(mean(temp_result),4), round(mean(time),4)];
                    row = row+1;
                    result(row,:) = [0, 0, 0, 0 round(std(temp_result,0,1),4), 0];
                    row = row+1;  
                end
            end
        end
    end
    file_addr = strcat("./para_best/", data_name);
    save(file_addr, "result");
end




