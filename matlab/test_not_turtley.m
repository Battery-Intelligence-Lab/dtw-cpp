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

figure; spy(sol_mip.w)

%%

sol_penalized_isCluster = findBestMedoids(distanceMat, Nc, 2, true);

%%

sol_mip2 = findBestMedoids(distanceMat, Nc, 2); %% Lower gap! 

%%

sol_turtle2 = findBestMedoids(distanceMat, Nc, 2, true);

%%

sol_turtle_Mosek = findBestMedoids(distanceMat, Nc, 2, true);

%%

