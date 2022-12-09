% This is a test function. Please do not use it. 

clear variables; close all; clc; 


Nc = 10;

distanceMat = readmatrix('AllGestureWiimoteX_TEST_distanceMatrix.csv');
assert(size(distanceMat,1) == size(distanceMat,2)); % See if it is square

tic; 
sol_mip = findBestMedoids(distanceMat, Nc, 2);

toc

fprintf('Cost: %4.6f\n',sol.cost);

%%

sol_turtle = findBestMedoids(distanceMat, Nc, 2, true);

%%
Nb = length(distanceMat);
maybe_clusters = find(sol_turtle.isCluster>0.01);
N_maybe = length(maybe_clusters);
F = []; % Constraints

w = sdpvar(N_maybe, Nb, 'full');
F = [F, 0 <= w <= 1];

isCluster = sdpvar(N_maybe,1);
F = [F, 0 <= isCluster <= 1];


F = [F, sum(w,1) == 1]; % Only one cluster can be assigned.
F = [F, w <= repmat(isCluster,1,Nb)]; % if w of ith data is activated then it is a cluster.
F = [F, sum(isCluster)== Nc];  % There should be Nc clusters.

cost =  sum(w.*distanceMat(maybe_clusters,:),'all');

optimize(F,cost, sdpsettings('verbose',2,'gurobi.MIPGap',1e-6, 'gurobi.NumericFocus',3));

%%

sol_reduced.w = value(w);
sol_reduced.isCluster = value(isCluster);
sol_reduced.cost = value(cost);

distance_reduced = distanceMat(maybe_clusters,:);

%%
deffo_clusters = find(sol_turtle.isCluster>0.9);
deffo_clusters_reduced = find(sol_reduced.isCluster>0.9);

sum(sol_reduced.w(deffo_clusters_reduced,:).*distance_reduced(deffo_clusters_reduced,:),'all')
sum(sol_turtle.w(deffo_clusters,:).*distanceMat(deffo_clusters,:),'all')

%%
N = Nb;
distanceMat2 = distanceMat(1:end,1:end) + 0.1*rand(size(distanceMat(1:end,1:end)));

lower = [zeros(N^2+N,1); zeros(N^2,1); ones(N,1); Nc];
upper = [ones(N^2+N,1);  ones(N^2,1); ones(N,1); Nc];

A = [speye(N^2+N); [kron(speye(N),-speye(N)), repmat(speye(N),N,1)];
     kron(speye(N+1), ones(1,N))];

x = sdpvar(N*(N+1), 1);

F = [(lower) <= A*x,  A*x<= (upper)];

q = [distanceMat2(:); zeros(N,1)];

J = q'*x + x(end-N:end)'*x(end-N:end);

optimize(F,J);

x_val = value(x);

w = reshape(x_val(1:N^2), N,N);
isCentroid =x_val(N^2+1:end);

%% 

bad_xval = ~(x_val<=0.01 | x_val>=0.99);
bad_A = A(any(A(:,bad_xval),2),:);
%%

[I, J] = find( ~(sol_turtle.w<=0.01 | sol_turtle.w>=0.99));

%%

X = zeros(N,N);
X(I,J) = 1;

spy(X)

%% New problem: 

[I, J] = find( distanceMat2 <1e-5);



