%> @file example_quickstart.m
%> @brief DTWC++ Quickstart Example — DTW distance, distance matrix, and clustering.
%> @author Volkan Kumtepeli
%% DTWC++ Quickstart Example
% Compute DTW distance, build a distance matrix, and cluster time series.
%
% Prerequisites:
%   Build the MEX file: cmake .. -DDTWC_BUILD_MATLAB=ON && cmake --build .
%   Ensure dtwc_mex is on the MATLAB path.

%% 1. Pairwise DTW distance
x = sin(linspace(0, 2*pi, 100));
y = cos(linspace(0, 2*pi, 100));

d = dtwc.dtw_distance(x, y);
fprintf('DTW distance (sin vs cos): %.4f\n', d);

% Banded DTW — constrain warping to speed up computation
% @author Volkan Kumtepeli
d_banded = dtwc.dtw_distance(x, y, 'Band', 10);
fprintf('DTW distance (band=10):    %.4f\n', d_banded);

%% 2. Distance matrix — computed in C++ with OpenMP
rng(42);  % reproducibility
N = 20;
L = 100;
data = randn(N, L);

dm = dtwc.compute_distance_matrix(data);
fprintf('\nDistance matrix: %dx%d\n', size(dm));
fprintf('Min non-zero: %.4f\n', min(dm(dm > 0)));
fprintf('Max:          %.4f\n', max(dm(:)));
fprintf('Symmetric:    %d\n', issymmetric(dm));

%% 3. Clustering — k-medoids with FastPAM
clust = dtwc.DTWClustering('NClusters', 3, 'Band', 10);
labels = clust.fit_predict(data);

fprintf('\nCluster labels (1-based):\n');
disp(labels);

fprintf('Cluster sizes: ');
for k = 1:3
    fprintf('%d ', sum(labels == k));
end
fprintf('\n');
fprintf('Total cost: %.2f\n', clust.TotalCost);
fprintf('Medoid indices: ');
disp(clust.MedoidIndices);
