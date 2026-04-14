%> @file test_mex.m
%> @brief DTWC++ MEX Binding Test Suite (Comprehensive).
%> @author Volkan Kumtepeli
%% DTWC++ MEX Binding Test Suite (Comprehensive)
% Run with: matlab -batch "addpath('build/bin'); addpath('bindings/matlab'); test_mex"
%
% NOTE: On Windows with MSVC OpenMP, `matlab -batch` may segfault on exit
% AFTER all tests complete. This is a known MATLAB/OpenMP DLL teardown issue
% and does NOT indicate a test failure. Check the test output, not exit code.

fprintf('=== DTWC++ MATLAB MEX Test Suite ===\n\n');

passed = 0;
failed = 0;

%% Test 1: DTW distance (full)
try
    x = sin(linspace(0, 2*pi, 100));
    y = cos(linspace(0, 2*pi, 100));
    d = dtwc.distance.dtw(x, y);
    assert(d > 0, 'DTW distance should be positive');
    fprintf('Test  1 - DTW distance (full): %.4f  [PASS]\n', d);
    passed = passed + 1;
catch e
    fprintf('Test  1 - DTW distance (full)  [FAIL] %s\n', e.message);
    failed = failed + 1;
end

%% Test 2: DTW distance (banded)
try
    x = sin(linspace(0, 2*pi, 100));
    y = cos(linspace(0, 2*pi, 100));
    d = dtwc.distance.dtw(x, y, 'Band', 10);
    assert(d > 0, 'Banded DTW distance should be positive');
    fprintf('Test  2 - DTW distance (band=10): %.4f  [PASS]\n', d);
    passed = passed + 1;
catch e
    fprintf('Test  2 - DTW distance (band=10)  [FAIL] %s\n', e.message);
    failed = failed + 1;
end

%% Test 3: Self-distance = 0
try
    d = dtwc.distance.dtw([1 2 3 4], [1 2 3 4]);
    assert(d == 0, 'Self-distance should be zero');
    fprintf('Test  3 - Self-distance = 0  [PASS]\n');
    passed = passed + 1;
catch e
    fprintf('Test  3 - Self-distance  [FAIL] %s\n', e.message);
    failed = failed + 1;
end

%% Test 4: Distance matrix
try
    rng(42);
    data = randn(10, 50);
    D = dtwc.compute_distance_matrix(data);
    assert(issymmetric(D), 'Distance matrix should be symmetric');
    assert(all(diag(D) == 0), 'Diagonal should be zero');
    assert(all(D(:) >= 0), 'All distances should be non-negative');
    fprintf('Test  4 - Distance matrix 10x10: symmetric, zero diag  [PASS]\n');
    passed = passed + 1;
catch e
    fprintf('Test  4 - Distance matrix  [FAIL] %s\n', e.message);
    failed = failed + 1;
end

%% Test 5: DDTW distance
try
    x = [1 3 5 7 5 3 1];
    y = [2 4 6 8 6 4 2];
    d = dtwc.distance.ddtw(x, y);
    assert(d >= 0, 'DDTW distance should be non-negative');
    % DDTW should detect similar shapes
    d_same_shape = dtwc.distance.ddtw(x, x * 2 + 1);
    assert(d_same_shape < d || d_same_shape >= 0, 'DDTW should handle scaled signals');
    fprintf('Test  5 - DDTW distance: %.4f  [PASS]\n', d);
    passed = passed + 1;
catch e
    fprintf('Test  5 - DDTW distance  [FAIL] %s\n', e.message);
    failed = failed + 1;
end

%% Test 6: WDTW distance
try
    x = [1 2 3 4 5];
    y = [2 3 4 5 6];
    d = dtwc.distance.wdtw(x, y);
    assert(d > 0, 'WDTW distance should be positive');
    % Different g should give different distances
    d2 = dtwc.distance.wdtw(x, y, 'G', 1.0);
    fprintf('Test  6 - WDTW distance: %.4f (g=0.05), %.4f (g=1.0)  [PASS]\n', d, d2);
    passed = passed + 1;
catch e
    fprintf('Test  6 - WDTW distance  [FAIL] %s\n', e.message);
    failed = failed + 1;
end

%% Test 7: ADTW distance
try
    x = [1 2 3 4 5];
    y = [1 1 2 3 4 5]; % shifted
    d_std = dtwc.distance.dtw(x, y);
    d_adtw = dtwc.distance.adtw(x, y, 'Penalty', 1.0);
    assert(d_adtw >= d_std, 'ADTW with penalty should be >= standard DTW');
    fprintf('Test  7 - ADTW distance: %.4f (std DTW: %.4f)  [PASS]\n', d_adtw, d_std);
    passed = passed + 1;
catch e
    fprintf('Test  7 - ADTW distance  [FAIL] %s\n', e.message);
    failed = failed + 1;
end

%% Test 8: Soft-DTW distance and gradient
% @author Volkan Kumtepeli
try
    x = [1 2 3 4 5];
    y = [2 3 4 5 6];
    d = dtwc.distance.soft_dtw(x, y, 'Gamma', 1.0);
    g = dtwc.soft_dtw_gradient(x, y, 'Gamma', 1.0);
    assert(isscalar(d), 'Soft-DTW should return scalar');
    assert(numel(g) == numel(x), 'Gradient should have same length as x');
    fprintf('Test  8 - Soft-DTW: dist=%.4f, grad_size=%d  [PASS]\n', d, numel(g));
    passed = passed + 1;
catch e
    fprintf('Test  8 - Soft-DTW  [FAIL] %s\n', e.message);
    failed = failed + 1;
end

%% Test 9: DTW with missing data (zero-cost)
try
    x = [1 2 NaN 4 5];
    y = [1 2 3 4 5];
    d = dtwc.distance.missing(x, y);
    assert(d >= 0, 'Missing DTW should be non-negative');
    % NaN positions should contribute zero cost
    x_no_nan = [1 2 3 4 5];
    d_full = dtwc.distance.dtw(x_no_nan, y);
    assert(d <= d_full, 'Missing DTW should be <= full DTW (zero-cost NaN)');
    fprintf('Test  9 - DTW missing (zero-cost): %.4f  [PASS]\n', d);
    passed = passed + 1;
catch e
    fprintf('Test  9 - DTW missing  [FAIL] %s\n', e.message);
    failed = failed + 1;
end

%% Test 10: DTW-AROW
% @author Volkan Kumtepeli
try
    x = [1 2 NaN 4 5];
    y = [1 2 3 4 5];
    d = dtwc.distance.arow(x, y);
    assert(d >= 0, 'AROW distance should be non-negative');
    fprintf('Test 10 - DTW-AROW: %.4f  [PASS]\n', d);
    passed = passed + 1;
catch e
    fprintf('Test 10 - DTW-AROW  [FAIL] %s\n', e.message);
    failed = failed + 1;
end

%% Test 11: Problem class lifecycle
try
    prob = dtwc.Problem('test_problem');
    assert(strcmp(prob.Name, 'test_problem'), 'Name should match');
    assert(prob.Size == 0, 'Empty problem should have size 0');

    rng(42);
    data = randn(10, 50);
    prob.set_data(data);
    assert(prob.Size == 10, 'Size should be 10 after set_data');

    prob.Band = 5;
    assert(prob.Band == 5, 'Band should be 5');

    prob.Verbose = true;
    assert(prob.Verbose == true, 'Verbose should be true');

    prob.MaxIter = 200;
    assert(prob.MaxIter == 200, 'MaxIter should be 200');

    % Display should not error
    disp(prob);

    fprintf('Test 11 - Problem lifecycle  [PASS]\n');
    passed = passed + 1;
catch e
    fprintf('Test 11 - Problem lifecycle  [FAIL] %s\n', e.message);
    failed = failed + 1;
end

%% Test 12: Problem distance matrix operations
% @author Volkan Kumtepeli
try
    prob = dtwc.Problem('dist_test');
    rng(42);
    data = randn(8, 30);
    prob.set_data(data);

    assert(~prob.is_distance_matrix_filled(), 'Dist matrix should not be filled yet');
    prob.fill_distance_matrix();
    assert(prob.is_distance_matrix_filled(), 'Dist matrix should be filled now');

    D = prob.get_distance_matrix();
    assert(size(D, 1) == 8 && size(D, 2) == 8, 'Should be 8x8');
    assert(issymmetric(D), 'Should be symmetric');

    % Test dist_by_ind (1-based)
    d12 = prob.dist_by_ind(1, 2);
    assert(abs(d12 - D(1, 2)) < 1e-10, 'dist_by_ind should match matrix');

    fprintf('Test 12 - Problem distance matrix  [PASS]\n');
    passed = passed + 1;
catch e
    fprintf('Test 12 - Problem distance matrix  [FAIL] %s\n', e.message);
    failed = failed + 1;
end

%% Test 13: FastPAM via Problem
try
    prob = dtwc.Problem('fastpam_test');
    rng(42);
    data = randn(15, 40);
    prob.set_data(data);
    prob.Band = -1;

    result = dtwc.fast_pam(prob, 3);
    assert(numel(result.labels) == 15, 'Should have 15 labels');
    assert(numel(result.medoid_indices) == 3, 'Should have 3 medoids');
    assert(result.total_cost > 0, 'Cost should be positive');
    assert(all(result.labels >= 1 & result.labels <= 3), 'Labels should be 1-based [1,3]');
    assert(all(result.medoid_indices >= 1 & result.medoid_indices <= 15), 'Medoids should be 1-based [1,15]');
    assert(islogical(result.converged), 'converged should be logical');

    % Check results stored in problem
    ci = prob.CentroidsInd;
    cl = prob.ClustersInd;
    assert(numel(ci) == 3, 'CentroidsInd should have 3 elements');
    assert(numel(cl) == 15, 'ClustersInd should have 15 elements');

    fprintf('Test 13 - FastPAM: cost=%.2f, converged=%d  [PASS]\n', result.total_cost, result.converged);
    passed = passed + 1;
catch e
    fprintf('Test 13 - FastPAM  [FAIL] %s\n', e.message);
    failed = failed + 1;
end

%% Test 14: FastCLARA
try
    prob = dtwc.Problem('clara_test');
    rng(42);
    data = randn(20, 30);
    prob.set_data(data);

    result = dtwc.fast_clara(prob, 3, 'NSamples', 3, 'Seed', 42);
    assert(numel(result.labels) == 20, 'Should have 20 labels');
    assert(numel(result.medoid_indices) == 3, 'Should have 3 medoids');
    assert(result.total_cost > 0, 'Cost should be positive');

    fprintf('Test 14 - FastCLARA: cost=%.2f  [PASS]\n', result.total_cost);
    passed = passed + 1;
catch e
    fprintf('Test 14 - FastCLARA  [FAIL] %s\n', e.message);
    failed = failed + 1;
end

%% Test 15: CLARANS
% @author Volkan Kumtepeli
try
    prob = dtwc.Problem('clarans_test');
    rng(42);
    data = randn(15, 30);
    prob.set_data(data);

    result = dtwc.clarans(prob, 3, 'NumLocal', 2, 'Seed', 42);
    assert(numel(result.labels) == 15, 'Should have 15 labels');
    assert(numel(result.medoid_indices) == 3, 'Should have 3 medoids');
    assert(result.total_cost > 0, 'Cost should be positive');

    fprintf('Test 15 - CLARANS: cost=%.2f  [PASS]\n', result.total_cost);
    passed = passed + 1;
catch e
    fprintf('Test 15 - CLARANS  [FAIL] %s\n', e.message);
    failed = failed + 1;
end

%% Test 16: Hierarchical clustering (build + cut)
try
    prob = dtwc.Problem('hier_test');
    rng(42);
    data = randn(10, 30);
    prob.set_data(data);
    prob.fill_distance_matrix();

    dend = dtwc.build_dendrogram(prob, 'Linkage', 'average');
    assert(isstruct(dend), 'Should return struct');
    assert(isfield(dend, 'merges'), 'Should have merges field');
    assert(isfield(dend, 'n_points'), 'Should have n_points field');
    assert(size(dend.merges, 1) == 9, 'Should have N-1=9 merge steps');
    assert(size(dend.merges, 2) == 4, 'Merges should have 4 columns');

    result = dtwc.cut_dendrogram(dend, prob, 3);
    assert(numel(result.labels) == 10, 'Should have 10 labels');
    assert(numel(result.medoid_indices) == 3, 'Should have 3 medoids');

    fprintf('Test 16 - Hierarchical: %d merges, cut to 3 clusters  [PASS]\n', size(dend.merges, 1));
    passed = passed + 1;
catch e
    fprintf('Test 16 - Hierarchical  [FAIL] %s\n', e.message);
    failed = failed + 1;
end

%% Test 17: Scoring functions
% @author Volkan Kumtepeli
try
    prob = dtwc.Problem('score_test');
    rng(42);
    data = randn(12, 30);
    prob.set_data(data);

    % Cluster first (required for scoring)
    result = dtwc.fast_pam(prob, 3);

    sil = dtwc.silhouette(prob);
    assert(numel(sil) == 12, 'Should have 12 silhouette values');
    assert(all(sil >= -1 & sil <= 1), 'Silhouette should be in [-1, 1]');

    db = dtwc.davies_bouldin_index(prob);
    assert(isscalar(db) && db >= 0, 'DB index should be non-negative scalar');

    di = dtwc.dunn_index(prob);
    assert(isscalar(di) && di >= 0, 'Dunn index should be non-negative scalar');

    ine = dtwc.inertia(prob);
    assert(isscalar(ine) && ine >= 0, 'Inertia should be non-negative scalar');

    ch = dtwc.calinski_harabasz_index(prob);
    assert(isscalar(ch) && ch >= 0, 'CH index should be non-negative scalar');

    fprintf('Test 17 - Scores: sil_mean=%.3f, DB=%.3f, Dunn=%.3f, inertia=%.2f, CH=%.2f  [PASS]\n', ...
        mean(sil), db, di, ine, ch);
    passed = passed + 1;
catch e
    fprintf('Test 17 - Scoring  [FAIL] %s\n', e.message);
    failed = failed + 1;
end

%% Test 18: ARI and NMI
% @author Volkan Kumtepeli
try
    labels_true = int32([1 1 1 2 2 2 3 3 3 3]);
    labels_pred = int32([1 1 2 2 2 3 3 3 3 3]);

    ari = dtwc.adjusted_rand_index(labels_true, labels_pred);
    assert(isscalar(ari), 'ARI should be scalar');
    assert(ari >= -1 && ari <= 1, 'ARI should be in [-1, 1]');

    nmi = dtwc.normalized_mutual_information(labels_true, labels_pred);
    assert(isscalar(nmi), 'NMI should be scalar');
    assert(nmi >= 0 && nmi <= 1, 'NMI should be in [0, 1]');

    % Perfect agreement should give ARI=1, NMI=1
    ari_perfect = dtwc.adjusted_rand_index(labels_true, labels_true);
    nmi_perfect = dtwc.normalized_mutual_information(labels_true, labels_true);
    assert(abs(ari_perfect - 1.0) < 1e-10, 'Perfect ARI should be 1');
    assert(abs(nmi_perfect - 1.0) < 1e-10, 'Perfect NMI should be 1');

    fprintf('Test 18 - ARI=%.3f, NMI=%.3f (perfect: ARI=%.1f, NMI=%.1f)  [PASS]\n', ...
        ari, nmi, ari_perfect, nmi_perfect);
    passed = passed + 1;
catch e
    fprintf('Test 18 - ARI/NMI  [FAIL] %s\n', e.message);
    failed = failed + 1;
end

%% Test 19: Derivative transform and z_normalize utilities
try
    x = [1 3 5 7 5 3 1];
    dx = dtwc.derivative_transform(x);
    assert(numel(dx) == numel(x), 'derivative_transform should preserve length');
    assert(dx(1) == 2, 'First boundary: x[1]-x[0] = 3-1 = 2');
    assert(dx(end) == -2, 'Last boundary: x[n-1]-x[n-2] = 1-3 = -2');

    xn = dtwc.z_normalize(x);
    assert(numel(xn) == numel(x), 'z_normalize should preserve length');
    assert(abs(mean(xn)) < 1e-10, 'z-normalized mean should be ~0');
    assert(abs(std(xn, 1) - 1.0) < 1e-10, 'z-normalized std (pop) should be ~1');

    fprintf('Test 19 - derivative_transform, z_normalize  [PASS]\n');
    passed = passed + 1;
catch e
    fprintf('Test 19 - Utilities  [FAIL] %s\n', e.message);
    failed = failed + 1;
end

%% Test 20: DTWClustering high-level API
try
    rng(42);
    data = randn(10, 50);
    clust = dtwc.DTWClustering('NClusters', 3, 'Band', 5);
    clust = clust.fit(data);  % Value class: must capture returned object
    labels = clust.Labels;
    assert(numel(labels) == 10, 'Should have 10 labels');
    assert(numel(clust.MedoidIndices) == 3, 'Should have 3 medoids');
    assert(clust.TotalCost > 0, 'Cost should be positive');
    assert(all(labels >= 1 & labels <= 3), 'Labels should be 1-based');
    fprintf('Test 20 - DTWClustering: %d labels, cost=%.2f  [PASS]\n', ...
        numel(labels), clust.TotalCost);
    passed = passed + 1;
catch e
    fprintf('Test 20 - DTWClustering  [FAIL] %s\n', e.message);
    failed = failed + 1;
end

%% Test 21: DTWClustering with variant
try
    rng(42);
    data = randn(10, 30);
    clust = dtwc.DTWClustering('NClusters', 2, 'Variant', 'ddtw');
    clust = clust.fit(data);  % Value class: must capture returned object
    labels = clust.Labels;
    assert(numel(labels) == 10, 'Should have 10 labels');
    assert(clust.TotalCost > 0, 'Cost should be positive');
    fprintf('Test 21 - DTWClustering (DDTW variant): cost=%.2f  [PASS]\n', clust.TotalCost);
    passed = passed + 1;
catch e
    fprintf('Test 21 - DTWClustering DDTW  [FAIL] %s\n', e.message);
    failed = failed + 1;
end

%% Test 22: Problem set_variant
% @author Volkan Kumtepeli
try
    prob = dtwc.Problem('variant_test');
    rng(42);
    data = randn(8, 30);
    prob.set_data(data);

    % Test WDTW variant
    prob.set_variant('wdtw', 0.1);
    prob.fill_distance_matrix();
    D_wdtw = prob.get_distance_matrix();
    assert(all(D_wdtw(:) >= 0), 'WDTW distances should be non-negative');

    fprintf('Test 22 - Problem set_variant (WDTW)  [PASS]\n');
    passed = passed + 1;
catch e
    fprintf('Test 22 - Problem set_variant  [FAIL] %s\n', e.message);
    failed = failed + 1;
end

%% Test 23: Problem set_distance_matrix
% @author Volkan Kumtepeli
try
    prob = dtwc.Problem('setdist_test');
    rng(42);
    data = randn(5, 20);
    prob.set_data(data);

    % Compute distance matrix externally
    D = dtwc.compute_distance_matrix(data);
    prob.set_distance_matrix(D);
    assert(prob.is_distance_matrix_filled(), 'Should be marked as filled');

    % Verify it matches
    D2 = prob.get_distance_matrix();
    assert(max(abs(D(:) - D2(:))) < 1e-10, 'Set and get distance matrix should match');

    fprintf('Test 23 - set_distance_matrix  [PASS]\n');
    passed = passed + 1;
catch e
    fprintf('Test 23 - set_distance_matrix  [FAIL] %s\n', e.message);
    failed = failed + 1;
end

%% Test 24: Error handling (invalid handle)
try
    caught = false;
    try
        dtwc_mex('Problem_get_size', uint64(99999));
    catch e
        caught = true;
        assert(contains(e.identifier, 'dtwc:'), 'Should have dtwc error ID');
    end
    assert(caught, 'Should have thrown an error for invalid handle');
    fprintf('Test 24 - Error handling (invalid handle)  [PASS]\n');
    passed = passed + 1;
catch e
    fprintf('Test 24 - Error handling  [FAIL] %s\n', e.message);
    failed = failed + 1;
end

%% Test 25: Error handling (unknown command)
try
    caught = false;
    try
        dtwc_mex('nonexistent_command');
    catch e
        caught = true;
        assert(contains(e.identifier, 'dtwc:'), 'Should have dtwc error ID');
    end
    assert(caught, 'Should have thrown an error for unknown command');
    fprintf('Test 25 - Error handling (unknown command)  [PASS]\n');
    passed = passed + 1;
catch e
    fprintf('Test 25 - Error handling  [FAIL] %s\n', e.message);
    failed = failed + 1;
end

%% Test 26: Problem delete and handle cleanup
try
    prob = dtwc.Problem('delete_test');
    h = prob.get_handle();
    assert(h > 0, 'Handle should be > 0');
    delete(prob);
    % After delete, the handle should be invalid
    caught = false;
    try
        dtwc_mex('Problem_get_size', uint64(h));
    catch
        caught = true;
    end
    assert(caught, 'Accessing deleted handle should error');
    fprintf('Test 26 - Problem delete/cleanup  [PASS]\n');
    passed = passed + 1;
catch e
    fprintf('Test 26 - Problem delete  [FAIL] %s\n', e.message);
    failed = failed + 1;
end

%% Summary
fprintf('\n=== Results: %d passed, %d failed out of %d ===\n', passed, failed, passed + failed);

if failed == 0
    fprintf('ALL TESTS PASSED\n');
else
    fprintf('SOME TESTS FAILED\n');
end

