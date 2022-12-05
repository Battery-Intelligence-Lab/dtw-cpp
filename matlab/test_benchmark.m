% This file is to compare results of benchmarks against our results. 
clear all; close all; clc;

file = '../data/benchmark/UCRArchive_2018/UMD/UMD_TEST.tsv';

data = readmatrix(file,'FileType','text', 'Delimiter','\t');

distanceMat = readmatrix("UMD_test_matrix.csv"); % DTW distances calculated by C++ 
%%

Nc = data(end,1); % Number of clusters. 

clusters = cell(Nc,1);
costs = zeros(Nc,1);
medoids = zeros(Nc,1);
for i=1:Nc
    clusters{i} = find(data(:,1)==i);
    cost_mat = distanceMat(clusters{i}, clusters{i});

    possible_cost = sum(cost_mat);

    [min_cost, min_k] = min(possible_cost);

    costs(i) = min_cost;
    medoids(i) = clusters{i}(min_k);
end

total_cost = sum(costs);

fprintf('Total cost: %4.6f\n',total_cost);
fprintf('Medoids: %d\n', medoids);


%%

tic; 
sol = findBestMedoids(distanceMat, Nc, 2);

toc

fprintf('Cost: %4.6f\n',sol.cost);
