function tests = test_contract_parity
%TEST_CONTRACT_PARITY MATLAB-column parity gate for docs/api-contract-2.0.md.
%
%   Asserts that EVERY symbol in the MATLAB column of the FROZEN API contract
%   (docs/api-contract-2.0.md) is present and callable through the +dtwc package
%   / dtwc_mex gateway. This is the Phase 2 Task 2.2 parity gate: it drives the
%   real public entry points (Tier-1 device/load/cluster/Result, Tier-2 Problem
%   setters+methods, scores, algorithms, distance functions, and the §2.7
%   checkpoint surface), not dead siblings.
%
%   Each test comment names the contract section and the public entry point it
%   exercises (LESSONS: tests pin the live code path).
%
%   Run with: results = runtests('test_contract_parity');
%   (Requires the compiled dtwc_mex on the path; otherwise every test is SKIPPED
%    with a loud notice.)
    tests = functiontests(localfunctions);
end

% -------------------------------------------------------------------------
%  Fixtures
% -------------------------------------------------------------------------

function setupOnce(testCase)
    testCase.TestData.mex_available = (exist('dtwc_mex', 'file') == 3); % 3 == MEX-file
    if ~testCase.TestData.mex_available
        bar = repmat('=', 1, 74);
        warning('dtwc:mexNotBuilt', ['\n' bar '\n' ...
            'SKIPPING dtwc contract-parity tests.\n' ...
            'Reason: compiled gateway ''dtwc_mex'' not found on the MATLAB path.\n' ...
            'Build with: cmake -B build -DDTWC_BUILD_MATLAB=ON && cmake --build build\n' ...
            'then addpath(build bin) and addpath(''bindings/matlab'').\n' bar]);
    end
    rng(42);
    % Two obvious clusters (rows = series), even length so ndim=2 is valid.
    testCase.TestData.X = [ones(3, 8); 10 * ones(3, 8)] + 0.01 * randn(6, 8);
    testCase.TestData.k = 2;
    % Deterministic global device for every test.
    if testCase.TestData.mex_available
        dtwc.device('cpu');
    end
end

function setup(testCase)
    assumeTrue(testCase, testCase.TestData.mex_available, ...
        'dtwc_mex MEX unavailable - skipping (see loud warning above).');
end

function prob = make_filled_problem(testCase)
%MAKE_FILLED_PROBLEM Helper: a Problem with data + filled distance matrix + a fit.
    prob = dtwc.Problem('parity');
    prob.set_data(testCase.TestData.X);
    prob.fill_distance_matrix();
    dtwc.fast_pam(prob, testCase.TestData.k);   % writes labels/medoids back
end

% =========================================================================
%  Tier 1 (contract §1)
% =========================================================================

function test_tier1_device_get_set(testCase)
%   §1.1 dtwc.device -> MEX set_device/get_device -> dtwc::Env.
    name = dtwc.device('cpu');
    verifyEqual(testCase, name, 'cpu');
    verifyEqual(testCase, dtwc.device(), 'cpu');
end

function test_tier1_device_unknown_raises_deviceError(testCase)
%   §5/§6 no silent fallback: an unknown device name -> dtwc:deviceError.
    verifyError(testCase, @() dtwc.device('definitely_not_a_device'), 'dtwc:deviceError');
end

function test_tier1_load_matrix_and_options(testCase)
%   §1.2 dtwc.load -> dtwc.Dataset (lazy handle, no I/O for a matrix source).
    ds = dtwc.load(testCase.TestData.X, 'skip_cols', 0, 'delimiter', '', 'name', 'ptest');
    verifyClass(testCase, ds, 'dtwc.Dataset');
    verifyEqual(testCase, ds.Name, 'ptest');
    [Xm, ~] = ds.materialize();
    verifySize(testCase, Xm, size(testCase.TestData.X));
end

function test_tier1_cluster_returns_result(testCase)
%   §1.3 dtwc.cluster -> §1.4 dtwc.Result (Tier-1 pam path).
    res = dtwc.cluster(testCase.TestData.X, testCase.TestData.k, ...
                       'method', 'pam', 'band', -1, 'device', '', 'max_iter', 50);
    verifyClass(testCase, res, 'dtwc.Result');
    verifyNumElements(testCase, res.labels, 6);
    verifyNumElements(testCase, res.medoids, 2);
    verifyGreaterThanOrEqual(testCase, res.cost, 0);
    verifyEqual(testCase, res.device, 'cpu');
end

function test_tier1_default_seed_matches_cpp_and_python(testCase)
%   M13: ambiguous waveforms expose 29-vs-42 initialization/local-optimum drift.
    X = (0:7)' + [0 0.01 -0.02 0.03];
    verifyEqual(testCase, dtwc.default_random_seed(), 42);

    tier1 = dtwc.cluster(X, 3, 'method', 'pam');

    prob42 = dtwc.Problem('seed42');
    prob42.set_data(X);
    seeded42 = dtwc.fast_pam(prob42, 3, 'Seed', 42);

    prob29 = dtwc.Problem('seed29');
    prob29.set_data(X);
    seeded29 = dtwc.fast_pam(prob29, 3, 'Seed', 29);

    verifyEqual(testCase, seeded42.medoid_indices, int32([7 3 6]));
    verifyEqual(testCase, seeded42.labels, int32([2 2 2 2 3 3 1 1]));
    verifyEqual(testCase, seeded42.total_cost, 24);
    verifyEqual(testCase, seeded29.medoid_indices, int32([5 2 8]));
    verifyEqual(testCase, seeded29.labels, int32([2 2 2 1 1 1 3 3]));
    verifyEqual(testCase, seeded29.total_cost, 20);

    verifyEqual(testCase, tier1.medoids, seeded42.medoid_indices);
    verifyEqual(testCase, tier1.labels, seeded42.labels);
    verifyEqual(testCase, tier1.cost, seeded42.total_cost);
    verifyNotEqual(testCase, tier1.medoids, seeded29.medoid_indices);
end

function test_dtwclustering_restarts_use_distinct_local_seeds(testCase)
%   NInit=2 must try seeds 42 and 43; repeating 42 makes the second run useless.
    X = (0:7)' + [0 0.01 -0.02 0.03];
    one = dtwc.DTWClustering('NClusters', 3, 'NInit', 1);
    one = one.fit(X);
    two = dtwc.DTWClustering('NClusters', 3, 'NInit', 2);
    two = two.fit(X);
    verifyEqual(testCase, one.TotalCost, 24);
    verifyEqual(testCase, two.TotalCost, 20);
    verifyLessThan(testCase, two.TotalCost, one.TotalCost);
end

function test_dtwclustering_metric_routes_match_exhaustive_oracle(testCase)
%   F18: DTWClustering.fit/fit_predict must execute the requested metric.
%   All three monotone paths for each length-two pair were enumerated before
%   implementation; the diagonal is uniquely optimal because every added
%   off-diagonal local cost is strictly positive.
    X = [0 1; 3 8; 5 2; 6 4];
    D_l1 = [0 10 6 9; 10 0 8 7; 6 8 0 3; 9 7 3 0];
    D_squared = [0 58 26 45; 58 0 40 25; ...
                 26 40 0 5; 45 25 5 0];

    dtwc.device('cpu');
    routed_l1 = dtwc_mex('DTWClustering_compute_distance_matrix', ...
                         double(X), -1, 'l1');
    routed_squared = dtwc_mex('DTWClustering_compute_distance_matrix', ...
                              double(X), -1, 'squared_euclidean');
    assertEqual(testCase, routed_l1, D_l1);
    assertEqual(testCase, routed_squared, D_squared);

    l1_problem = dtwc.Problem('F18_l1_oracle');
    l1_problem.set_data(X);
    l1_problem.set_distance_matrix(D_l1);
    l1_result = dtwc.fast_pam(l1_problem, 2, 'MaxIter', 100, 'Seed', 42);
    assertEqual(testCase, l1_result.labels, int32([1 2 1 1]));
    assertEqual(testCase, l1_result.medoid_indices, int32([3 2]));
    assertEqual(testCase, l1_result.total_cost, 9);
    assertEqual(testCase, l1_result.iterations, int32(1));
    assertTrue(testCase, l1_result.converged);

    squared_problem = dtwc.Problem('F18_squared_oracle');
    squared_problem.set_data(X);
    squared_problem.set_distance_matrix(D_squared);
    squared_result = dtwc.fast_pam(squared_problem, 2, ...
                                   'MaxIter', 100, 'Seed', 42);
    assertEqual(testCase, squared_result.labels, int32([2 1 1 1]));
    assertEqual(testCase, squared_result.medoid_indices, int32([4 1]));
    assertEqual(testCase, squared_result.total_cost, 30);
    assertEqual(testCase, squared_result.iterations, int32(1));
    assertTrue(testCase, squared_result.converged);

    l1_estimator = dtwc.DTWClustering( ...
        'NClusters', 2, 'Metric', 'l1', 'Device', 'cpu', 'NInit', 2);
    l1_estimator = l1_estimator.fit(X);
    assertEqual(testCase, l1_estimator.Labels, int32([1 2 1 1]));
    assertEqual(testCase, l1_estimator.MedoidIndices, int32([3 2]));
    assertEqual(testCase, l1_estimator.TotalCost, 9);

    squared_estimator = dtwc.DTWClustering( ...
        'NClusters', 2, 'Metric', 'squared_euclidean', ...
        'Device', 'cpu', 'NInit', 2);
    squared_estimator = squared_estimator.fit(X);
    assertEqual(testCase, squared_estimator.Labels, int32([2 1 1 1]));
    assertEqual(testCase, squared_estimator.MedoidIndices, int32([4 1]));
    assertEqual(testCase, squared_estimator.TotalCost, 30);

    uppercase_estimator = dtwc.DTWClustering( ...
        'NClusters', 2, 'Metric', 'SQUARED_EUCLIDEAN', ...
        'Device', 'cpu', 'NInit', 2);
    uppercase_labels = uppercase_estimator.fit_predict(X);
    assertEqual(testCase, uppercase_labels, int32([2 1 1 1]));

    assertNotEqual(testCase, l1_estimator.Labels, squared_estimator.Labels);
    assertNotEqual(testCase, l1_estimator.MedoidIndices, ...
                   squared_estimator.MedoidIndices);
    assertNotEqual(testCase, l1_estimator.TotalCost, ...
                   squared_estimator.TotalCost);

    fprintf(['F18_MATLAB_METRIC subject=DTWClustering.fit+fit_predict ' ...
        'oracle=exhaustive_paths matrices=2/2 problem_routes=2/2 ' ...
        'fit_routes=2/2 fit_predict=1/1 case_norm=1/1 distinct=3/3 ' ...
        'ninit=2 skips=0\n']);
end

function test_dtwclustering_metric_validation_precedes_effects(testCase)
%   F18: invalid metric/cross-product requests fail before data/device work.
    X = [0 1; 3 8; 5 2; 6 4];
    dtwc.device('cpu');

    unknown_error = [];
    unknown = dtwc.DTWClustering( ...
        'NClusters', 2, 'Metric', 'not_a_metric', ...
        'Device', 'not_a_device', 'NInit', 2);
    try
        unknown.fit(zeros(0, 2));
    catch caught
        unknown_error = caught;
    end
    assertFalse(testCase, isempty(unknown_error), ...
                'Unknown Metric must fail before empty-data/device handling.');
    assertEqual(testCase, unknown_error.identifier, 'dtwc:invalidArgument');
    assertEqual(testCase, unknown_error.message, ...
        ['Unknown Metric ''not_a_metric''. Expected one of: ' ...
         'l1, squared_euclidean.']);
    assertEqual(testCase, dtwc.device(), 'cpu');

    variant_error = [];
    bad_variant = dtwc.DTWClustering( ...
        'NClusters', 2, 'Metric', 'squared_euclidean', ...
        'Device', 'cpu', 'NInit', 2);
    bad_variant.Variant = 'wdtw';
    try
        bad_variant.fit(X);
    catch caught
        variant_error = caught;
    end
    assertFalse(testCase, isempty(variant_error), ...
                'SquaredL2 plus WDTW must fail loudly.');
    assertEqual(testCase, variant_error.identifier, 'dtwc:invalidArgument');

    missing_error = [];
    bad_missing = dtwc.DTWClustering( ...
        'NClusters', 2, 'Metric', 'squared_euclidean', ...
        'Device', 'cpu', 'NInit', 2);
    bad_missing.MissingStrategy = 'zero_cost';
    try
        bad_missing.fit(X);
    catch caught
        missing_error = caught;
    end
    assertFalse(testCase, isempty(missing_error), ...
                'SquaredL2 plus zero_cost must fail loudly.');
    assertEqual(testCase, missing_error.identifier, 'dtwc:invalidArgument');

    fprintf(['F18_MATLAB_VALIDATION unknown_metric=1/1 ' ...
        'unknown_precedence=1/1 squared_variant=1/1 ' ...
        'squared_missing=1/1 skips=0\n']);
end

function test_fast_pam_mex_rejects_invalid_seed_before_cast(testCase)
%   Direct gateway calls are defensive even when bypassing inputParser.
    prob = dtwc.Problem('invalid_seed');
    prob.set_data((0:3)' + [0 0.01 -0.02 0.03]);
    call = @(seed) dtwc_mex('fast_pam', prob.get_handle(), 2, 100, seed);
    verifyError(testCase, @() call(-1), 'dtwc:invalidArgument');
    verifyError(testCase, @() call(NaN), 'dtwc:invalidArgument');
    verifyError(testCase, @() call(flintmax + 1), 'dtwc:invalidArgument');
end

function test_tier1_cluster_unknown_method_raises(testCase)
%   §1.3 unknown method -> dtwc:invalidArgument (never silently PAM).
    verifyError(testCase, ...
        @() dtwc.cluster(testCase.TestData.X, 2, 'method', 'no_such_method'), ...
        'dtwc:invalidArgument');
end

function test_tier1_result_score_names(testCase)
%   §1.4 Result.score(name) for every accepted score name.
    res = dtwc.cluster(testCase.TestData.X, testCase.TestData.k);
    for nm = {'silhouette', 'davies_bouldin', 'dunn', 'calinski_harabasz', 'inertia'}
        s = res.score(nm{1});
        verifyTrue(testCase, isscalar(s) && isnumeric(s), ...
            sprintf('score(%s) must return a numeric scalar', nm{1}));
    end
    verifyError(testCase, @() res.score('nope'), 'dtwc:invalidArgument');
end

function test_tier1_result_save_and_plot(testCase)
%   §1.4 Result.save(dir) writes the 4 CSVs; Result.plot() renders (headless).
    res = dtwc.cluster(testCase.TestData.X, testCase.TestData.k);
    outdir = fullfile(tempdir, ['dtwc_parity_' num2str(feature('getpid'))]);
    res.save(outdir);
    nm = 'dataset';
    for suffix = {'_labels.csv', '_medoids.csv', '_distance_matrix.csv', '_silhouettes.csv'}
        f = fullfile(outdir, [nm suffix{1}]);
        verifyTrue(testCase, isfile(f), sprintf('save() must emit %s', suffix{1}));
    end
    ax = res.plot();
    verifyTrue(testCase, isa(ax, 'matlab.graphics.axis.Axes'));
    close all force;
end

function test_tier1_dtwclustering_device_param(testCase)
%   §1.5 DTWClustering gains a Device parameter (delegates to Env).
    c = dtwc.DTWClustering('NClusters', 2, 'Device', 'cpu');
    verifyEqual(testCase, c.Device, 'cpu');
    c = c.fit(testCase.TestData.X);
    verifyNumElements(testCase, c.Labels, 6);
end

% =========================================================================
%  Tier 2 — Problem config setters (contract §2.1)
% =========================================================================

function test_problem_setters_all_callable(testCase)
%   §2.1 every MATLAB-column setter on dtwc.Problem.
    prob = dtwc.Problem('setters');
    prob.set_data(testCase.TestData.X);
    prob.set_n_clusters(2);
    prob.set_method('kmedoids');
    prob.set_band(3);
    prob.set_max_iter(50);
    prob.set_n_repetitions(1);
    prob.set_variant('wdtw', 0.1);
    prob.set_variant('standard');
    prob.set_missing_strategy('error');
    prob.set_distance_strategy('auto');
    prob.set_lb_strategy('keogh');
    prob.set_storage_policy('heap');
    ok = prob.set_solver('highs');
    verifyTrue(testCase, islogical(ok));
    prob.set_output_folder(tempdir);
    prob.set_verbose(false);
    prob.set_cuda_settings(0, 0);
    verifyEqual(testCase, prob.size(), 6);
end

function test_problem_enhanced_webb_lb_strategies(testCase)
%   Every core lower-bound strategy is selectable through the MATLAB parser.
    prob = dtwc.Problem('lb_strategy_parity');
    prob.set_lb_strategy('enhanced');
    prob.set_lb_strategy('webb');
end

function test_problem_set_mip_settings_roundtrip(testCase)
%   §2.1 set_mip_settings(struct) + get_mip_settings (MIPSettings + benders).
    prob = dtwc.Problem('mip');
    s = struct('mip_gap', 1e-4, 'time_limit_sec', 30, 'warm_start', true, ...
               'max_benders_iter', 150, 'benders', 'on');
    prob.set_mip_settings(s);
    got = prob.get_mip_settings();
    verifyEqual(testCase, got.mip_gap, 1e-4, 'AbsTol', 1e-12);
    verifyEqual(testCase, got.max_benders_iter, 150);
    verifyEqual(testCase, char(got.benders), 'on');
end

function test_problem_set_data_ragged(testCase)
%   §2.1 data (owning): ragged input via a cell array of numeric vectors.
    prob = dtwc.Problem('ragged');
    C = {[1 2 3 4], [1 2 3 4 5 6], [9 9 9]};
    prob.set_data(C);
    verifyEqual(testCase, prob.size(), 3);
end

function test_problem_set_data_names(testCase)
%   §2.1 series names alongside the data matrix.
    prob = dtwc.Problem('named');
    names = {'a', 'b', 'c', 'd', 'e', 'f'};
    prob.set_data(testCase.TestData.X, names);
    verifyEqual(testCase, prob.size(), 6);
end

function test_problem_set_data_ndim_multivariate(testCase)
%   §2.1 ndim multivariate (interleaved layout); L must be divisible by ndim.
    prob = dtwc.Problem('mv');
    prob.set_data(testCase.TestData.X, {}, 2);   % 8 cols -> 4 timesteps x 2 features
    verifyEqual(testCase, prob.size(), 6);
    prob.fill_distance_matrix();
    verifyTrue(testCase, prob.is_distance_matrix_filled());
end

% =========================================================================
%  Tier 2 — Problem distance-matrix & clustering methods (contract §2.2)
% =========================================================================

function test_problem_methods_all_callable(testCase)
%   §2.2 refresh/read/max_distance/dist_by_ind/fill/distance_matrix/set/cluster.
    prob = dtwc.Problem('methods');
    prob.set_data(testCase.TestData.X);
    prob.fill_distance_matrix();
    verifyTrue(testCase, prob.is_distance_matrix_filled());

    D = prob.distance_matrix();                 % canonical (rename of get_distance_matrix)
    verifySize(testCase, D, [6 6]);
    verifyEqual(testCase, prob.dist_by_ind(1, 2), D(1, 2), 'AbsTol', 1e-10);
    verifyGreaterThanOrEqual(testCase, prob.max_distance(), 0);

    % In-place clustering on the filled matrix. cluster() (Kmedoids/Lloyd) writes
    % per-rep medoid CSVs, so point the output folder at a writable temp dir.
    prob.set_output_folder(tempdir);
    prob.set_n_clusters(2);
    prob.cluster();
    verifyGreaterThanOrEqual(testCase, prob.find_total_cost(), 0);

    % Remaining §2.2 methods exercised as isolated callable live paths.
    prob.set_distance_matrix(D);
    prob.refresh_distance_matrix();             % clears cached matrix
    csvpath = fullfile(tempdir, 'dtwc_parity_D.csv');
    writematrix(D, csvpath);
    prob.read_distance_matrix(csvpath);         % CSV reader (swallows on format mismatch)
end

function test_problem_semantic_setters_invalidate_dense_cache(testCase)
%   Semantic setters must discard distances computed under the prior contract.
    prob = dtwc.Problem('semantic_mutation');
    prob.set_missing_strategy('zero_cost');
    prob.set_data([0 NaN 2; 0 2 2]);
    verifyEqual(testCase, prob.dist_by_ind(1, 2), 0, 'AbsTol', 0);

    prob.set_missing_strategy('interpolate');
    verifyFalse(testCase, prob.is_distance_matrix_filled());
    verifyEqual(testCase, prob.dist_by_ind(1, 2), 1, 'AbsTol', 1e-12);

    D = [0 123; 123 0];
    prob.set_distance_matrix(D);
    prob.set_distance_strategy('brute_force');
    verifyFalse(testCase, prob.is_distance_matrix_filled());

    prob.set_distance_matrix(D);
    prob.set_cuda_settings(3, 2);
    verifyFalse(testCase, prob.is_distance_matrix_filled());
end

function test_problem_read_accessors(testCase)
%   §2.2 read accessors: size / n_clusters / name / labels / medoids.
    prob = make_filled_problem(testCase);
    verifyEqual(testCase, prob.size(), 6);
    verifyEqual(testCase, prob.n_clusters(), 2);
    verifyEqual(testCase, prob.name(), 'parity');
    verifyNumElements(testCase, prob.labels(), 6);
    verifyNumElements(testCase, prob.medoids(), 2);
end

% =========================================================================
%  Tier 2 — scores (contract §2.4, canonical snake_case names)
% =========================================================================

function test_scores_canonical_names(testCase)
%   §2.4 silhouette/davies_bouldin/dunn/inertia/calinski_harabasz/adjusted_rand/
%        normalized_mutual_info.
    prob = make_filled_problem(testCase);
    verifyNumElements(testCase, dtwc.silhouette(prob), 6);
    verifyTrue(testCase, isscalar(dtwc.davies_bouldin(prob)));
    verifyTrue(testCase, isscalar(dtwc.dunn(prob)));
    verifyTrue(testCase, isscalar(dtwc.inertia(prob)));
    verifyTrue(testCase, isscalar(dtwc.calinski_harabasz(prob)));

    lt = int32([1 1 2 2 3 3]);
    lp = int32([1 1 2 2 3 3]);
    verifyEqual(testCase, dtwc.adjusted_rand(lt, lp), 1, 'AbsTol', 1e-12);
    verifyEqual(testCase, dtwc.normalized_mutual_info(lt, lp), 1, 'AbsTol', 1e-12);
end

% =========================================================================
%  Tier 2 — algorithm free functions (contract §2.5)
% =========================================================================

function test_algorithms_all_callable(testCase)
%   §2.5 fast_pam / fast_clara / clarans / build_dendrogram / cut_dendrogram.
    prob = dtwc.Problem('algos');
    prob.set_data(testCase.TestData.X);
    prob.fill_distance_matrix();

    r1 = dtwc.fast_pam(prob, 2);
    verifyNumElements(testCase, r1.labels, 6);
    r2 = dtwc.fast_clara(prob, 2, 'NSamples', 2, 'Seed', 42);
    verifyNumElements(testCase, r2.labels, 6);
    r3 = dtwc.clarans(prob, 2, 'NumLocal', 2, 'Seed', 42);
    verifyNumElements(testCase, r3.labels, 6);

    dend = dtwc.build_dendrogram(prob, 'Linkage', 'average');
    verifyTrue(testCase, isstruct(dend) && isfield(dend, 'merges'));
    r4 = dtwc.cut_dendrogram(dend, prob, 2);
    verifyNumElements(testCase, r4.labels, 6);
end

% =========================================================================
%  Tier 2 — distance free functions (contract §2.6)
% =========================================================================

function test_distance_functions_all_callable(testCase)
%   §2.6 dtwc.distance.{standard,ddtw,wdtw,adtw,soft_dtw,missing,arow,dtw}.
    x = [1 2 3 4 5];
    y = [2 3 4 5 6];
    verifyGreaterThanOrEqual(testCase, dtwc.distance.standard(x, y), 0);
    verifyGreaterThanOrEqual(testCase, dtwc.distance.ddtw(x, y), 0);
    verifyGreaterThanOrEqual(testCase, dtwc.distance.wdtw(x, y), 0);
    verifyGreaterThanOrEqual(testCase, dtwc.distance.adtw(x, y), 0);
    verifyTrue(testCase, isscalar(dtwc.distance.soft_dtw(x, y, 'Gamma', 1.0)));
    verifyGreaterThanOrEqual(testCase, dtwc.distance.missing([1 NaN 3 4 5], y), 0);
    verifyGreaterThanOrEqual(testCase, dtwc.distance.arow([1 NaN 3 4 5], y), 0);
    verifyGreaterThanOrEqual(testCase, dtwc.distance.dtw(x, y), 0);
    verifyGreaterThanOrEqual(testCase, dtwc.distance.dtw(x, y, 'Variant', 'wdtw', 'G', 0.1), 0);
end

% =========================================================================
%  Tier 2 — checkpoint / resume (contract §2.7)
% =========================================================================

function test_checkpoint_dir_roundtrip(testCase)
%   §2.7 CheckpointOptions / save_checkpoint / load_checkpoint.
    opts = dtwc.CheckpointOptions('directory', tempdir, 'enabled', true);
    verifyTrue(testCase, isstruct(opts) && islogical(opts.enabled));

    prob = dtwc.Problem('ckpt');
    prob.set_data(testCase.TestData.X);
    prob.fill_distance_matrix();
    ckdir = fullfile(tempdir, ['dtwc_parity_ck_' num2str(feature('getpid'))]);
    dtwc.save_checkpoint(prob, ckdir);

    prob2 = dtwc.Problem('ckpt2');
    prob2.set_data(testCase.TestData.X);
    ok = dtwc.load_checkpoint(prob2, ckdir);
    verifyTrue(testCase, islogical(ok) && ok);
end

function test_checkpoint_binary_roundtrip(testCase)
%   §2.7 save_binary_checkpoint / load_binary_checkpoint (ClusteringResult).
    prob = make_filled_problem(testCase);
    result = dtwc.fast_pam(prob, 2);
    binpath = fullfile(tempdir, ['dtwc_parity_' num2str(feature('getpid')) '.bin']);
    dtwc.save_binary_checkpoint(result, binpath);
    loaded = dtwc.load_binary_checkpoint(binpath);
    verifyEqual(testCase, numel(loaded.labels), numel(result.labels));
    verifyEqual(testCase, sort(double(loaded.medoid_indices)), ...
                          sort(double(result.medoid_indices)));
end
