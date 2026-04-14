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
    verifyLessThanOrEqual(testCase, max(c.MedoidIndices), 8);
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
    verifySize(testCase, labels, [1, 6]);
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

