function [ind_res,num_c] = get_incomplete(num_sample,per,num_view)  %num_c是完整的样本数
    %input:per为缺失数据百分比
    %outout:一个num_sample*num_view的指示矩阵，数值为0的位置为缺失数据index
    ind_folds = [];
    record = zeros(1,num_sample);   %用于记录每一列被去除的次数
    for v=1:num_view
        empty_temp = randperm(num_sample,floor(num_sample*per));
        for j = 1:floor(num_sample*per)
            ind_folds(j,v) = empty_temp(j);
            record(1,empty_temp(j)) = record(1,empty_temp(j))+1;
        end
        ind_folds(:,v) = sort(ind_folds(:,v));
    end

    %需要确保不会出现某一列数据在所有视图中都缺失的情况：
    allMis = [];   %用于记录被全部缺失的列
    prepare = [];   %备选的替换列
    index_allMis = 1;   %allMis的指针
    index_prepare = 1;   %prepare的指针
    for k = 1:size(record,2)
        if record(k) == num_view
            allMis(index_allMis) = k;
            index_allMis = index_allMis +1;
        end
        if record(k) <= num_view-2
            prepare(index_prepare) = k;
            index_prepare = index_prepare + 1;
        end
    end
    
    num = size(allMis,2);  %记录全被去除的列的个数
    if num ~= 0   %如果出现了某一列被全部去除的情况
        for i = 1:num   %遍历每一个全去列
            num_pre = size(prepare,2);

            rand_view = randi([1, num_view]);   %更改随机某个视图中的对应的全去列
            for m = 1:size(prepare,2)   %遍历每一个可选的列
                if (~ismember(prepare(1,m),ind_folds(:,rand_view)))
                    rand_sample = prepare(1, m);   %将其更改为备选列中的某一列
                    break
                end
            end
            for j = m:size(prepare,2)-1   %将使用的备选列从prepare中去除
                prepare(j) = prepare(j+1);
            end
            prepare(num_pre) = [];

            for k = 1:size(ind_folds,1)
                if ind_folds(k,rand_view) == allMis(i)
                    ind_folds(k,rand_view) = rand_sample;
                end
            end
        end
    end
    %还要重新排序
    for v=1:num_view
        ind_folds(:,v) = sort(ind_folds(:,v));
    end

    %检验是否不存在全被去除的列
    num_c = 0;
    record_new = zeros(1,num_sample);   %用于记录每一列被去除的次数（新）
    check = 0;
    for v=1:num_view
        for j = 1:floor(num_sample*per)
            record_new(1,ind_folds(j,v)) = record_new(1,ind_folds(j,v))+1;
        end
    end
    for k = 1:size(record,2)
        if record_new(k) == num_view
            check = check+1;
        end
        if record_new(k) == 0
            num_c = num_c+1;
        end
    end

    ind_res = ones(num_sample, num_view);
    for i = 1:size(ind_folds,2)
        for j = 1:size(ind_folds,1)
            ind_res(ind_folds(j,i),i) = 0;
        end
    end
end


