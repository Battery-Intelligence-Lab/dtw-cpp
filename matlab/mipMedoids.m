% This file is for finding k-Medoids using integer programming. 
% Author: Vk
% Date: 2022.02.22


clear all; close all; clc;


distanceMat = readmatrix('../results/DTWdist_band_all.csv');
assert(size(distanceMat,1) == size(distanceMat,2)); % See if it is square

normalizedDistance = distanceMat/mean(distanceMat(:));

explanation = '1077 cells, band 2000';

tic;
for Nc=11:50

fprintf('Nc = %d, sol is started.\n',Nc);
sol = findBestMedoids(normalizedDistance, Nc);

fprintf('Nc = %d, sol square is started.\n',Nc);
sol_squared = findBestMedoids(normalizedDistance.^2, Nc);

save(['solve_2022_02_28_Nc_',num2str(Nc),'_.mat'],'sol','sol_squared','distanceMat','explanation');

end
toc