function [Zor] = GenerateZ_2(fea,num_anchor,ind_folds)
num_view = length(fea);
% num_sample= length(fea{1});
Zo = cell(1, num_view);
Zor= cell(1, num_view);
% [~, Anchors1] = litekmeans(fea{1}', num_anchor);
paried_index=ones(size(fea{1},2),1);
% for t=1:num_view
%     paried_index= paried_index.*ind_folds(:,t);
% end
% for t=1:num_view
%     anchor_temp = fea{t};
%     unparied_index = find(paried_index == 0);
%     anchor_temp(:,unparied_index) = []; 
%     Anchor{t}= anchor_temp;
% end

for t=1:num_view
    fea_temp=fea{t};
    ind_0 = find(ind_folds(:,t) == 0);
    fea_temp(:,ind_0) = []; 
    [~, Anchors1] = litekmeans(fea_temp', num_anchor);
    Zo{t} = full(ConstructA_NP(fea_temp, Anchors1'));
    Zo{t}=Zo{t}';
    G = diag(ind_folds(:,t));
    G(:,ind_0) = [];
    Zor{t} = Zo{t}*G';
end
end
