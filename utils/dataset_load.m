function [X,Y] = dataset_load(data_addr, data_name)
    load(data_addr);

%     if strcmp(data_name, 'MSRC_v1')==1
%        X = fea;
%        Y = gt;
%     end

    if strcmp(data_name, 'HW')==1 || strcmp(data_name, 'cifar10')==1 || strcmp(data_name, 'BBCSport')==1
       X = data;
       Y = truelabel{1};
    end

    if strcmp(data_name, 'MSRC_v1')==1
        X = fea;
        Y = gt;
    end

    if strcmp(data_name, '3sources3vbigRnSp')==1
        Y = truth;
    end


%     if strcmp(data_name, 'BBCSport')==1
%        X = data;
%        Y = truelabel{1};
%     end

    if size(X{1},1) ~= size(Y)
        for i = 1:length(X)
            X{i} = X{i}';
        end 
    else
        for i = 1:length(X)
            X{i} = X{i};
        end
    end
   
end

