function sol = findBestMedoids(distanceMat, Nc, verbosity)

if(nargin<3)
    verbosity = 0;
end

Nb = length(distanceMat); % Number of batteries. 


is_w_binVar = false;
is_isCluster_binVar = false;
F = []; % Constraints

if(is_w_binVar)
w = binvar(Nb, Nb, 'full');
else
w = sdpvar(Nb, Nb, 'full');
F = [F, 0 <= w <= 1];
end


if(is_isCluster_binVar)
isCluster = binvar(Nb,1);
else
isCluster = sdpvar(Nb,1); 
F = [F, 0 <= isCluster <= 1];
end


F = [F, sum(w,1) == 1]; % Only one cluster can be assigned. 
F = [F, w <= repmat(isCluster,1,Nb)]; % if w of ith data is activated then it is a cluster. 
F = [F, sum(isCluster)== Nc];  % There should be Nc clusters. 





cost =  sum(w.*distanceMat,'all');

yalmipStr = optimize(F,cost,sdpsettings('verbose',verbosity,'gurobi.MIPGap',1e-6));

%%
sol.yalmipStr = yalmipStr;
sol.w = value(w);
sol.isCluster = value(isCluster);
sol.cost = sum(value(w).* distanceMat,'all');

sol.clusters = find(sol.isCluster);

assert(length(sol.clusters)==Nc);


sol.members = {};

for i_cluster = 1:Nc
    sol.members{i_cluster} = find(sol.w(sol.clusters(i_cluster),:));
end


end