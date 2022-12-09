% This file is to compare results of benchmarks against our results. 
clear all; close all; clc;

% file = '../data/benchmark/UCRArchive_2018/UMD/UMD_TEST.tsv';
% 
% data = readmatrix(file,'FileType','text', 'Delimiter','\t');

distanceMat = readmatrix("TwoPatterns_TEST_distanceMatrix.csv"); % DTW distances calculated by C++ 
Nc = 4;
%%

tic; 
sol = findBestMedoids(distanceMat, Nc, 2);

toc

fprintf('Cost: %4.6f\n',sol.cost);
