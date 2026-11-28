function tests = test_dtwc
%TEST_DTWC Unit tests for the dtwc MATLAB package.
%   Run with: results = runtests('test_dtwc');

%> @file test_dtwc.m
%> @brief Unit tests for the dtwc MATLAB package.
%> @author Volkan Kumtepeli
%> @date 29 Mar 2026
    tests = functiontests(localfunctions);
end

%% --- dtwc.distance.dtw tests ---

function test_dtw_distance_column_vectors(testCase)
%TEST_DTW_DISTANCE_COLUMN_VECTORS Column vector inputs should work.
    x = [1; 2; 3];
    y = [3; 4; 5; 6; 7];
    verifyEqual(testCase, dtwc.distance.dtw(x, y), 13);
end

%% --- Distance, estimator, Problem and score properties ---

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

function test_softdtw_predict_reads_the_divergence_of_the_fit(testCase)
%   The flat medoid's sdtw(m, m) is far below the rough one's: raw Soft-DTW would put
%   the last series with the flat group, the divergence the fit clusters on with the rough.
    X = [zeros(1, 6); 0.05 zeros(1, 5); 0 0.05 zeros(1, 4); repmat([0 3], 1, 3); ...
         0.05 3 repmat([0 3], 1, 2); 0 2.95 repmat([0 3], 1, 2); repmat([0 2], 1, 3)];
    c = dtwc.DTWClustering('NClusters', 2, 'Variant', 'softdtw').fit(X);
    verifyEqual(testCase, double(c.predict(X)), double(c.Labels));
    verifyEqual(testCase, c.Labels(7), c.Labels(4));

    % SdtwGamma is validated and forwarded to fit and transform Soft-DTW divergence
    c_gamma = dtwc.DTWClustering('NClusters', 2, 'Variant', 'softdtw', 'SdtwGamma', 0.5).fit(X);
    D_t = c_gamma.transform(X);
    for i = 1:size(X, 1)
        for m = 1:numel(c_gamma.MedoidIndices)
            med = X(c_gamma.MedoidIndices(m), :);
            div = dtwc.distance.dtw(X(i, :), med, 'Variant', 'softdtw', 'SdtwGamma', 0.5) ...
                - 0.5 * (dtwc.distance.dtw(X(i, :), X(i, :), 'Variant', 'softdtw', 'SdtwGamma', 0.5) ...
                + dtwc.distance.dtw(med, med, 'Variant', 'softdtw', 'SdtwGamma', 0.5));
            verifyEqual(testCase, D_t(i, m), max(div, 0), 'AbsTol', 1e-12);
        end
    end
    verifyError(testCase, @() dtwc.DTWClustering('NClusters', 2, 'Variant', 'softdtw', 'SdtwGamma', 0).fit(X), ...
                'dtwc:invalidArgument');
    verifyError(testCase, @() dtwc.DTWClustering('NClusters', 2, 'Variant', 'softdtw', 'SdtwGamma', -1).fit(X), ...
                'dtwc:invalidArgument');
end

function test_problem_config_reaches_the_native_problem(testCase)
%TEST_PROBLEM_CONFIG_REACHES_THE_NATIVE_PROBLEM The setters write C++; Band, Verbose,
%   MaxIter and NRepetitions, and disp, read it back.
    prob = dtwc.Problem('config');
    verifyEqual(testCase, prob.size(), 0);
    prob.set_band(5);
    prob.set_verbose(true);
    prob.set_max_iter(200);
    prob.set_n_repetitions(3);
    verifyEqual(testCase, [prob.Band, prob.MaxIter, prob.NRepetitions], [5 200 3]);
    verifyTrue(testCase, prob.Verbose);
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

function test_a_float32_matrix_comes_back_single(testCase)
%   MatrixDtype 'float32' stores each distance rounded to a single, and the
%   matrix comes back at that width.
    rng(42);
    data = randn(5, 20);
    D = dtwc.compute_distance_matrix(data);
    S = dtwc.compute_distance_matrix(data, 'MatrixDtype', 'float32');
    verifyClass(testCase, S, 'single');
    verifyEqual(testCase, S, single(D));
    prob = dtwc.Problem('narrow');
    prob.set_matrix_dtype('f32');
    prob.set_data(data);
    prob.set_distance_matrix(S);
    verifyTrue(testCase, prob.is_distance_matrix_filled());
    verifyEqual(testCase, prob.distance_matrix(), S);
    prob.set_distance_matrix(D); % a double D is rounded as it is stored
    verifyEqual(testCase, prob.distance_matrix(), S);
    res = dtwc.cluster(data, 2, 'MatrixDtype', 'float32');
    verifyClass(testCase, res.distance_matrix(), 'single');
end

function test_a_negative_entry_is_refused_and_a_nan_one_computed(testCase)
%   v1.0.0 wrote -1 for a pair it never computed; 2.0 marks one NaN. A negative
%   entry is refused with the way to mark the pair: no distance is negative.
%   The oracle of a computed pair is dtwc.distance.dtw, the per-pair route.
    X = [0 1 3 2; 1 2 0 1; 4 0 2 2];
    D = dtwc.compute_distance_matrix(X);
    D(1, 2) = -1; D(2, 1) = -1;
    prob = dtwc.Problem('not_computed');
    prob.set_data(X);
    try
        prob.set_distance_matrix(D);
        verifyFail(testCase, 'a -1 entry was read as a distance');
    catch err
        verifyEqual(testCase, err.identifier, 'dtwc:invalidArgument');
        verifySubstring(testCase, err.message, 'is -1; no distance is negative');
        verifySubstring(testCase, err.message, 'D(D < 0) = NaN');
    end
    D(D < 0) = NaN;
    prob.set_distance_matrix(D);
    verifyFalse(testCase, prob.is_distance_matrix_filled());
    verifyEqual(testCase, prob.dist_by_ind(1, 2), dtwc.distance.dtw(X(1, :), X(2, :)));

    % A v1.0.0 file reads with 'NotComputed', 'negative'.
    path = [tempname '.csv'];
    cleanup = onCleanup(@() delete(path));
    fid = fopen(path, 'w');
    fprintf(fid, '0,-1,2.5\n-1,0,-1\n2.5,-1,0\n');
    fclose(fid);
    v1 = dtwc.Problem('v1');
    v1.set_data(X);
    verifyError(testCase, @() v1.read_distance_matrix(path), 'dtwc:invalidArgument');
    v1.read_distance_matrix(path, 'NotComputed', 'negative');
    verifyFalse(testCase, v1.is_distance_matrix_filled());
    verifyEqual(testCase, v1.dist_by_ind(1, 3), 2.5);
    verifyEqual(testCase, v1.dist_by_ind(2, 3), dtwc.distance.dtw(X(2, :), X(3, :)));
end

function test_wdtw_variant_fills_a_nonnegative_matrix(testCase)
    rng(42);
    prob = dtwc.Problem('wdtw');
    prob.set_data(randn(8, 30));
    prob.set_variant('wdtw', 0.1);
    prob.fill_distance_matrix();
    verifyGreaterThanOrEqual(testCase, prob.distance_matrix(), 0);
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
