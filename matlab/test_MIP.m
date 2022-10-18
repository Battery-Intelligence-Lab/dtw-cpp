% This is a test function. Please do not use it. 

clear variables; close all; clc; 


distanceMat = readmatrix('../results/DTW_matrix.csv');
assert(size(distanceMat,1) == size(distanceMat,2)); % See if it is square

sol = findBestMedoids(distanceMat, 4);

fprintf('Cost: %4.6f\n',sol.cost);