function [B,L,D,S,runtime1] = GenerateB(fea, num_anchor)
tic
cfea = [];
for t=1:length(fea)
    cfea = [fea{t},cfea];
end
fea{t+1} = cfea;
num_view = length(fea);
num_sample= length(fea{1});
B = cell(1, num_view);
L = cell(1, num_view);
S = cell(1, num_view);
D = cell(1, num_view);
for t=1:num_view
    [~, Anchors1] = kmeansplusplus(fea{t}', num_anchor);
    B{t} = ConstructA_NP(fea{t}', Anchors1);
    B{t}=B{t}';
    S{t}=B{t}*B{t}';
    D{t}=diag(sum(S{t},2));
    L{t}=D{t}-S{t};
end
runtime1 = toc;
end

