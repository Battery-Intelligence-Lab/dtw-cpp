% Hypothesis: Diagonal part of w shows the isCluster. Since a matrix has to
% belong its own cluster, only clusters has 1 at diagonal part. So
% isCluster matrix is on the diagonal 



F = []; % Constraints

w = sdpvar(Nb, Nb, 'full');
F = [F, 0 <= w <= 1];

F = [F, sum(w,1) == 1]; % Only one cluster can be assigned.
F = [F, w <= repmat(diag(w),1,Nb)]; % if w of ith data is activated then it is a cluster.
F = [F, sum(diag(w))== Nc];  % There should be Nc clusters.

cost =  sum(w.*distanceMat,'all');

optimize(F,cost, sdpsettings('verbose',2,'gurobi.MIPGap',1e-6, 'gurobi.NumericFocus',3));


%%
F = []; % Constraints


isCluster = sdpvar(Nb,1);
F = [F, 0 <= isCluster <= 1];

w = sdpvar(Nb, Nb, 'full');
F = [F, 0 <= w <= 1];

F = [F, sum(w,1) == 1]; % Only one cluster can be assigned.
F = [F, w <= repmat(isCluster,1,Nb)]; % if w of ith data is activated then it is a cluster.
F = [F, sum(isCluster)== Nc];  % There should be Nc clusters.


cost =  sum(w.*distanceMat,'all');

optimize(F,cost, sdpsettings('verbose',2,'gurobi.MIPGap',1e-6, 'gurobi.NumericFocus',3));


%% binvar w

F = []; % Constraints

w = binvar(Nb, Nb, 'full');

F = [F, sum(w,1) == 1]; % Only one cluster can be assigned.
F = [F, w <= repmat(diag(w),1,Nb)]; % if w of ith data is activated then it is a cluster.
F = [F, sum(diag(w))== Nc];  % There should be Nc clusters.

cost =  sum(w.*distanceMat,'all');

optimize(F,cost, sdpsettings('verbose',2,'gurobi.MIPGap',1e-6, 'gurobi.NumericFocus',3));


%% binvar isCluster
F = []; % Constraints

isCluster = binvar(Nb,1);
w = binvar(Nb, Nb, 'full');

F = [F, sum(w,1) == 1]; % Only one cluster can be assigned.
F = [F, w <= repmat(isCluster,1,Nb)]; % if w of ith data is activated then it is a cluster.
F = [F, sum(isCluster)== Nc];  % There should be Nc clusters.


cost =  sum(w.*distanceMat,'all');

optimize(F,cost, sdpsettings('verbose',2,'gurobi.MIPGap',1e-6, 'gurobi.NumericFocus',3));

