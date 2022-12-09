% This is a test function. Please do not use it. 

clear variables; close all; clc; 


distanceMat = readmatrix('DTW_dist_10.csv');
assert(size(distanceMat,1) == size(distanceMat,2)); % See if it is square

tic; 
sol = findBestMedoids(distanceMat, 4, 2);

toc

fprintf('Cost: %4.6f\n',sol.cost);


%% OSQP formulation
%N_dist = 10;
distanceMat2 = distanceMat(1:N_dist,1:N_dist);

Nc = 4;
N = size(distanceMat2,1); 

lower = [zeros(N^2+N,1); zeros(N^2,1); ones(N,1); Nc];
upper = [ones(N^2+N,1);  ones(N^2,1); ones(N,1); Nc];

A = [eye(N^2+N); [kron(eye(N),-eye(N)), repmat(eye(N),N,1)];
     kron(eye(N+1), ones(1,N))];

x = sdpvar(N*(N+1), 1);

F = [(lower) <= A*x,  A*x<= (upper)];

q = [distanceMat2(:); zeros(N,1)];

J = q'*x;


optimize(F,J);

x_val = value(x);

w = reshape(x_val(1:N^2), N,N)
isCentroid =x_val(N^2+1:end)

