function sol = findBestMedoids(distanceMat, Nc)
Nb = length(distanceMat); % Number of batteries. 
w = binvar(Nb, Nb, 'full');

isCluster = binvar(Nb,1); 

F = []; % Constraints
F = [F, sum(w,1) == 1]; % Only one cluster can be assigned. 
F = [F, w <= repmat(isCluster,1,Nb)]; % if w of ith data is activated then it is a cluster. 

F = [F, sum(isCluster)== Nc];  % There should be Nc clusters. 
%F = [F, 0 <= isCluster <= 1];

cost =  sum(w.*distanceMat,'all');

yalmipStr = optimize(F,cost,sdpsettings('verbose',0,'gurobi.MIPGap',1e-6));

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