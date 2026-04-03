%% DTWC++ MATLAB Cross-Language Benchmark
% Run with: matlab -batch "addpath('build/bin','bindings/matlab'); run('benchmarks/bench_cross_language.m')"

fprintf('=================================================================\n');
fprintf('DTWC++ MATLAB Benchmark\n');
fprintf('=================================================================\n');

%% DTW distance: 100 pairs
for L = [100, 500, 1000]
    rng(42);
    series_a = randn(100, L);
    series_b = randn(100, L);

    % Warm up
    dtwc.dtw_distance(series_a(1,:), series_b(1,:));

    tic;
    for i = 1:100
        dtwc.dtw_distance(series_a(i,:), series_b(i,:));
    end
    t = toc;
    fprintf('  DTW distance  100 pairs x L=%4d: %8.2f ms  (%.3f ms/pair)\n', L, t*1000, t/100*1000);
end

fprintf('\n');

%% Distance matrix
configs = {20, 100; 50, 100; 50, 500; 100, 500};
for c = 1:size(configs, 1)
    N = configs{c, 1};
    L = configs{c, 2};
    rng(42);
    data = randn(N, L);

    % Warm up (small)
    if N > 5
        dtwc.compute_distance_matrix(data(1:5, :));
    end

    tic;
    D = dtwc.compute_distance_matrix(data);
    t = toc;
    pairs = N * (N - 1) / 2;
    fprintf('  Dist matrix   %3dx%4d (%6d pairs): %8.2f ms\n', N, L, pairs, t*1000);
end

fprintf('\n');

%% Clustering (FastPAM)
cluster_configs = {20, 100, 3; 50, 100, 5; 50, 500, 5};
for c = 1:size(cluster_configs, 1)
    N = cluster_configs{c, 1};
    L = cluster_configs{c, 2};
    k = cluster_configs{c, 3};
    rng(42);
    data = randn(N, L);

    prob = dtwc.Problem('bench');
    prob.set_data(data);
    prob.Band = -1;

    tic;
    result = dtwc.fast_pam(prob, k);
    t = toc;
    fprintf('  FastPAM       %3dx%4d k=%d: %8.2f ms  (cost=%.2f)\n', N, L, k, t*1000, result.total_cost);
end

fprintf('\n');
fprintf('=================================================================\n');
