function [U,H, Obj] = EIMCAGC(Zor,num_cluster, beta,gamma,lambda,ind_folds)
num_sample = size(Zor{1}, 2);
num_anchor = size(Zor{1}, 1);
num_view = size(Zor, 2);
tol = 1e-4;
max_iter = 50;
tol_iter = 2;
% initialization
Zo=Zor;
% H = initializeH(num_sample,num_cluster);
fea=[];
for i=1:num_view
    fea=[fea,Zor{i}'];
end
[Indtemp, ~] = litekmeans(fea,num_cluster,'MaxIter', 100,'Replicates',10);
H= sparse(1:num_sample,Indtemp,1,num_sample,num_cluster,num_sample);
for i=1:num_view
    U{i} = initializeH(num_anchor,num_cluster);
end
alpha =1/num_view*ones(1,num_view);
B = rand(num_view,num_view);   % 初始化所有视角的重构稀疏权重为1
B = B-diag(diag(B)); % 去除自表示的影响
for m = 1:num_view
    indx = [1:num_view];
    indx(m) = [];
    B(indx',m) = (ProjSimplex(B(indx',m)'))'; % 将A的每一列按照列和归一化
end
for v = 1:num_view
     W{v} = ones(num_anchor,num_sample);
     ind_0 = find(ind_folds(:,v) == 0);  % indexes of misssing instances
     W{v}(:,ind_0) = 0;
end
delta = 1e-3;
for iter = 1 : max_iter
    fprintf('----processing iter %d--------\n', iter);
    %% Update U_v
    for v=1:num_view
        [UU, ~, WW] = svd(H'*Zor{v}','econ');
        U{v} = (UU*WW')';
    end
%     M
    %% Update H
    H_tem = zeros(num_sample, num_cluster);
    for v=1:num_view
    H_tem = H_tem + alpha(v)* EuDist2(Zor{v}', U{v}', 0);
    end
    [~, Indtemp] = min(H_tem, [], 2);
    H= sparse(1:num_sample,Indtemp,1,num_sample,num_cluster,num_sample);
    %% Update alpha
    r= zeros(1,num_view);
    vec_Z = [];
    for v = 1:num_view
       vec_Z = [vec_Z,(Zor{v}(:))];
    end    
    for v = 1:num_view
        r(v) = norm((Zor{v}-Zo{v}).*W{v},'fro')^2+lambda*norm(vec_Z(:,v)-vec_Z*B(:,v),'fro')^2+beta*trace((Zor{v}-U{v}*H')'*(Zor{v}-U{v}*H'));
        p(v) = double(exp(-r(v)/gamma)+eps);
    end
    alpha = p./(sum(p,2));

    %% Update B
    vec_Z = [];
    for v = 1:num_view
        vec_Z = [vec_Z,(Zor{v}(:))];
    end    
    for v = 1:num_view
        indv = [1:num_view];
        indv(v) = [];
        [B(indv',v),~] = SimplexRepresentation_acc(vec_Z(:,indv), vec_Z(:,v));
        %  min  || Ax - y||^2
        %  s.t. x>=0, 1'x=1
    end 
    %% Update Z
    vec_Z = [];
    for v = 1:num_view
        vec_Z = [vec_Z,(Zor{v}(:))];
    end
    for v=1:num_view
        P_v = vec_Z*B(:,v);
        P_v = reshape(P_v,num_anchor,num_sample);
        sum_Y = 0;
        coeef = 0;
        for v2 = 1:num_view
            if v2 ~= v
                sum_Y = sum_Y + alpha(v2)*B(v,v2)*lambda*(vec_Z(:,v2)-vec_Z*B(:,v2)+B(v,v2)*vec_Z(:,v));
                coeef = coeef +  B(v,v2)^2*alpha(v2);
            end
        end
        clear v2
        matrix_sum_Y = reshape(sum_Y,num_anchor,num_sample);
        clear sum_Y
        Linshi_L = (alpha(v)*Zo{v}.*W{v}+alpha(v)*lambda*P_v+alpha(v)*beta*U{v}*H'+matrix_sum_Y)./(alpha(v).*W{v}+lambda*alpha(v)+coeef*lambda+alpha(v)*beta);
        Zor{i}=Linshi_L;
        clear Linshi_L matrix_sum_Y coeef
    end

           %% Obj
    r= zeros(1,num_view);
    vec_Z = [];
    for v = 1:num_view
       vec_Z = [vec_Z,(Zor{v}(:))];
    end    
    for v = 1:num_view
        r(v) = norm((Zor{v}-Zo{v}).*W{v},'fro')^2+lambda*norm(vec_Z(:,v)-vec_Z*B(:,v),'fro')^2+beta*trace((Zor{v}-U{v}*H')'*(Zor{v}-U{v}*H'));
    end
   %% Obj
    r= zeros(1,num_view);
    vec_Z = [];
    for v = 1:num_view
       vec_Z = [vec_Z,(Zor{v}(:))];
    end    
    for v = 1:num_view
        r(v) = norm((Zor{v}-Zo{v}).*W{v},'fro')^2+lambda*norm(vec_Z(:,v)-vec_Z*B(:,v),'fro')^2+beta*trace((Zor{v}-U{v}*H')'*(Zor{v}-U{v}*H'));
    end
    Obj(iter) =alpha*r'+gamma*alpha*(log(alpha))'; 
    if iter > tol_iter &&  abs((Obj(iter) - Obj(iter-1) )/Obj(iter-1))< tol
        break; 
    end

end

end

