% This is a test function. Please do not use it. 

clear variables; close all; clc; 


distanceMat = readmatrix('AllGestureWiimoteX_TEST_distanceMatrix.csv');
assert(size(distanceMat,1) == size(distanceMat,2)); % See if it is square

tic; 
sol = findBestMedoids(distanceMat, 4, 2);

toc

fprintf('Cost: %4.6f\n',sol.cost);