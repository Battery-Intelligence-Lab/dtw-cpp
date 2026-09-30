function tests = test_dtwc
%TEST_DTWC Unit tests for the dtwc MATLAB package.
%   Run with: results = runtests('test_dtwc');
    tests = functiontests(localfunctions);
end

%% --- dtw_distance tests ---

function test_dtw_distance_basic(testCase)
%TEST_DTW_DISTANCE_BASIC Verify DTW distance is positive for different series.
    x = [1 2 3 4 5];
    y = [2 4 6 3 1];
    d = dtwc.distance.dtw(x, y);
    verifyGreaterThan(testCase, d, 0);
end

function test_dtw_distance_identical(testCase)
%TEST_DTW_DISTANCE_IDENTICAL Identical series should have zero distance.
    x = [1 2 3 4 5];
    d = dtwc.distance.dtw(x, x);
    verifyEqual(testCase, d, 0, 'AbsTol', 1e-12);
end

function test_dtw_distance_symmetric(testCase)
%TEST_DTW_DISTANCE_SYMMETRIC DTW distance should be symmetric.
    x = [1 3 5 2 4];
    y = [2 4 1 5 3];
    d1 = dtwc.distance.dtw(x, y);
    d2 = dtwc.distance.dtw(y, x);
    verifyEqual(testCase, d1, d2, 'AbsTol', 1e-12);
end

function test_dtw_distance_banded(testCase)
%TEST_DTW_DISTANCE_BANDED Banded DTW should be >= full DTW.
    x = [1 2 3 4 5 6 7 8 9 10];
    y = [2 4 6 8 10 9 7 5 3 1];
    d_full = dtwc.distance.dtw(x, y);
    d_band = dtwc.distance.dtw(x, y, 'Band', 2);
    verifyGreaterThanOrEqual(testCase, d_band, d_full - 1e-12);
end

function test_dtw_distance_unequal_length(testCase)
%TEST_DTW_DISTANCE_UNEQUAL_LENGTH DTW handles series of different lengths.
    x = [1 2 3];
    y = [1 2 3 4 5];
    d = dtwc.distance.dtw(x, y);
    verifyClass(testCase, d, 'double');
    verifyGreaterThanOrEqual(testCase, d, 0);
end

function test_dtw_distance_column_vectors(testCase)
%TEST_DTW_DISTANCE_COLUMN_VECTORS Column vector inputs should work.
    x = [1; 2; 3; 4; 5];
    y = [5; 4; 3; 2; 1];
    d = dtwc.distance.dtw(x, y);
    verifyGreaterThan(testCase, d, 0);
end

%% --- compute_distance_matrix tests ---

function test_distance_matrix_symmetric(testCase)
%TEST_DISTANCE_MATRIX_SYMMETRIC Output matrix should be symmetric.
    X = [1 2 3 4; 5 6 7 8; 1 3 5 7];
    D = dtwc.compute_distance_matrix(X);
    verifySize(testCase, D, [3 3]);
    verifyEqual(testCase, D, D', 'AbsTol', 1e-12);
end

function test_distance_matrix_diagonal_zero(testCase)
%TEST_DISTANCE_MATRIX_DIAGONAL_ZERO Diagonal must be zero.
    X = [1 2 3; 4 5 6; 7 8 9; 10 11 12];
    D = dtwc.compute_distance_matrix(X);
    verifyEqual(testCase, diag(D), zeros(4, 1), 'AbsTol', 1e-12);
end

function test_distance_matrix_nonneg(testCase)
%TEST_DISTANCE_MATRIX_NONNEG All entries must be non-negative.
    X = randn(5, 10);
    D = dtwc.compute_distance_matrix(X);
    verifyGreaterThanOrEqual(testCase, D, zeros(5));
end

%% --- DTWClustering tests ---

function test_clustering_basic(testCase)
%TEST_CLUSTERING_BASIC Labels should have correct dimensions and range.
    X = [ones(5,10); 10*ones(5,10)]; % two obvious clusters
    c = dtwc.DTWClustering('NClusters', 2);
    c = c.fit(X);
    verifySize(testCase, c.Labels, [1, 10]);
    verifyGreaterThanOrEqual(testCase, min(c.Labels), 1);
    verifyLessThanOrEqual(testCase, max(c.Labels), 2);
end

function test_clustering_medoids(testCase)
%TEST_CLUSTERING_MEDOIDS Medoid indices should be valid.
    X = [ones(5,8); 10*ones(5,8)];
    c = dtwc.DTWClustering('NClusters', 2);
    c = c.fit(X);
    verifySize(testCase, c.MedoidIndices, [1, 2]);
    verifyGreaterThanOrEqual(testCase, min(c.MedoidIndices), 1);
    verifyLessThanOrEqual(testCase, max(c.MedoidIndices), 10);
end

function test_clustering_cost_nonneg(testCase)
%TEST_CLUSTERING_COST_NONNEG Total cost should be non-negative.
    X = randn(5, 6);
    c = dtwc.DTWClustering('NClusters', 2);
    c = c.fit(X);
    verifyGreaterThanOrEqual(testCase, c.TotalCost, 0);
end

function test_clustering_fit_predict(testCase)
%TEST_CLUSTERING_FIT_PREDICT fit_predict should return labels directly.
    X = [ones(5,6); 10*ones(5,6)];
    c = dtwc.DTWClustering('NClusters', 2);
    labels = c.fit_predict(X);
    verifySize(testCase, labels, [1, 10]);
end

function test_clustering_constructor_defaults(testCase)
%TEST_CLUSTERING_CONSTRUCTOR_DEFAULTS Check default property values.
    c = dtwc.DTWClustering();
    verifyEqual(testCase, c.NClusters, 3);
    verifyEqual(testCase, c.Band, -1);
    verifyEqual(testCase, c.Metric, 'l1');
    verifyEqual(testCase, c.MaxIter, 100);
    verifyEqual(testCase, c.NInit, 1);
end

%% --- Distance, estimator, Problem and score properties ---

function test_adtw_penalty_never_undercuts_dtw(testCase)
%TEST_ADTW_PENALTY_NEVER_UNDERCUTS_DTW A penalty on off-diagonal steps cannot lower the cost.
    x = [1 2 3 4 5];
    y = [1 1 2 3 4 5];
    verifyGreaterThanOrEqual(testCase, ...
        dtwc.distance.adtw(x, y, 'Penalty', 1.0), dtwc.distance.dtw(x, y));
end

function test_zero_cost_missing_matches_the_filled_series(testCase)
%TEST_ZERO_COST_MISSING_MATCHES_THE_FILLED_SERIES A NaN costs nothing under 'missing'.
    y = [1 2 3 4 5];
    verifyEqual(testCase, dtwc.distance.missing([1 2 NaN 4 5], y), 0);
end

function test_soft_dtw_gradient_has_the_length_of_x(testCase)
%TEST_SOFT_DTW_GRADIENT_HAS_THE_LENGTH_OF_X One partial derivative per sample of x.
    x = [1 2 3 4 5];
    y = [2 3 4 5 6];
    verifyNumElements(testCase, dtwc.soft_dtw_gradient(x, y, 'Gamma', 1.0), numel(x));
end

function test_derivative_transform_boundaries(testCase)
%TEST_DERIVATIVE_TRANSFORM_BOUNDARIES The end points take the one-sided difference.
    x = [1 3 5 7 5 3 1];
    dx = dtwc.derivative_transform(x);
    verifyNumElements(testCase, dx, numel(x));
    verifyEqual(testCase, dx(1), 2);
    verifyEqual(testCase, dx(end), -2);
end

function test_z_normalize_has_zero_mean_and_unit_population_std(testCase)
    xn = dtwc.z_normalize([1 3 5 7 5 3 1]);
    verifyEqual(testCase, mean(xn), 0, 'AbsTol', 1e-10);
    verifyEqual(testCase, std(xn, 1), 1, 'AbsTol', 1e-10);
end

function test_clustering_variant_ddtw(testCase)
    rng(42);
    c = dtwc.DTWClustering('NClusters', 2, 'Variant', 'ddtw').fit(randn(10, 30));
    verifySize(testCase, c.Labels, [1, 10]);
    verifyGreaterThan(testCase, c.TotalCost, 0);
end

function test_problem_config_reaches_the_native_problem(testCase)
%TEST_PROBLEM_CONFIG_REACHES_THE_NATIVE_PROBLEM The setters write C++; disp reads it back.
    prob = dtwc.Problem('config');
    verifyEqual(testCase, prob.size(), 0);
    prob.set_band(5);
    prob.set_verbose(true);
    prob.set_max_iter(200);
    prob.set_n_repetitions(3);
    info = dtwc_mex('Problem_get_info', prob.get_handle());
    verifyEqual(testCase, [info.band, info.max_iter, info.n_repetitions], [5 200 3]);
    verifyTrue(testCase, info.verbose);
    verifySubstring(testCase, evalc('disp(prob)'), 'dtwc.Problem: "config"');
end

function test_set_distance_matrix_round_trips(testCase)
    rng(42);
    data = randn(5, 20);
    D = dtwc.compute_distance_matrix(data);
    prob = dtwc.Problem('setdist');
    prob.set_data(data);
    prob.set_distance_matrix(D);
    verifyTrue(testCase, prob.is_distance_matrix_filled());
    verifyEqual(testCase, prob.distance_matrix(), D, 'AbsTol', 1e-10);
end

function test_wdtw_variant_fills_a_nonnegative_matrix(testCase)
    rng(42);
    prob = dtwc.Problem('wdtw');
    prob.set_data(randn(8, 30));
    prob.set_variant('wdtw', 0.1);
    prob.fill_distance_matrix();
    verifyGreaterThanOrEqual(testCase, prob.distance_matrix(), 0);
end

function test_fast_pam_and_clara_results_are_one_based(testCase)
    rng(42);
    prob = dtwc.Problem('fastpam');
    prob.set_data(randn(15, 40));
    pam = dtwc.fast_pam(prob, 3);
    verifyTrue(testCase, all(pam.labels >= 1 & pam.labels <= 3));
    verifyTrue(testCase, all(pam.medoid_indices >= 1 & pam.medoid_indices <= 15));
    verifyNumElements(testCase, pam.medoid_indices, 3);
    verifyGreaterThan(testCase, pam.total_cost, 0);
    verifyClass(testCase, pam.converged, 'logical');
    verifyNumElements(testCase, prob.labels(), 15);
    verifyNumElements(testCase, prob.medoids(), 3);

    clara = dtwc.fast_clara(prob, 3, 'NSamples', 3, 'Seed', 42);
    verifyNumElements(testCase, clara.labels, 15);
    verifyGreaterThan(testCase, clara.total_cost, 0);
end

function test_dendrogram_has_n_minus_1_merges_of_4_columns(testCase)
    rng(42);
    prob = dtwc.Problem('hier');
    prob.set_data(randn(10, 30));
    prob.fill_distance_matrix();
    dend = dtwc.build_dendrogram(prob, 'Linkage', 'average');
    verifySize(testCase, dend.merges, [9 4]);
    verifyEqual(testCase, double(dend.n_points), 10);
    res = dtwc.cut_dendrogram(dend, prob, 3);
    verifyNumElements(testCase, res.labels, 10);
    verifyNumElements(testCase, res.medoid_indices, 3);
end

function test_scores_stay_in_their_ranges(testCase)
    rng(42);
    prob = dtwc.Problem('scores');
    prob.set_data(randn(12, 30));
    dtwc.fast_pam(prob, 3);
    sil = dtwc.silhouette(prob);
    verifyTrue(testCase, all(sil >= -1 & sil <= 1));
    verifyGreaterThanOrEqual(testCase, dtwc.davies_bouldin(prob), 0);
    verifyGreaterThanOrEqual(testCase, dtwc.dunn(prob), 0);
    verifyGreaterThanOrEqual(testCase, dtwc.inertia(prob), 0);
    verifyGreaterThanOrEqual(testCase, dtwc.calinski_harabasz(prob), 0);

    lt = int32([1 1 1 2 2 2 3 3 3 3]);
    lp = int32([1 1 2 2 2 3 3 3 3 3]);
    ari = dtwc.adjusted_rand(lt, lp);
    nmi = dtwc.normalized_mutual_info(lt, lp);
    verifyTrue(testCase, ari > -1 && ari < 1);
    verifyTrue(testCase, nmi > 0 && nmi < 1);
end
