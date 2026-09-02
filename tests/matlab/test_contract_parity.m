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

function test_tier1_load_skip_rows(testCase)
%   §1.2 skip_rows: header LINES for a path, leading SERIES for a matrix.
    ds = dtwc.load(testCase.TestData.X, 'skip_rows', 2);
    verifyEqual(testCase, ds.SkipRows, 2);
    [Xm, ~] = ds.materialize();
    verifySize(testCase, Xm, size(testCase.TestData.X) - [2 0]);

    f = [tempname '.csv'];
    fid = fopen(f, 'w');
    fprintf(fid, 'id,t0,t1\nunit,s,s\na,0,0\nb,10,11\n');
    fclose(fid);
    c = onCleanup(@() delete(f));
    hdr = dtwc.load(f, 'skip_cols', 1, 'skip_rows', 2, 'delimiter', ',');
    [Xh, ~] = hdr.materialize();
    verifyEqual(testCase, Xh, [0 0; 10 11]);
end

function test_tier1_load_negative_skip_rows_rejected(testCase)
%   §1.2 skip_rows is validated exactly as skip_cols is.
    cols = '';
    rows = '';
    try
        dtwc.load(testCase.TestData.X, 'skip_cols', -1);
    catch e
        cols = e.identifier;
    end
    try
        dtwc.load(testCase.TestData.X, 'skip_rows', -1);
    catch e
        rows = e.identifier;
    end
    verifyNotEmpty(testCase, cols);
    verifyEqual(testCase, rows, cols);

    % Both of the above are MATLAB's own inputParser rejections, so on their
    % own they would still pass if skip_rows were dropped downstream. Drive the
    % gateway directly to reach C++ detail::validate_skips and pin its verbatim
    % message (dtwc/api.cpp: "load: skip_rows must be non-negative.").
    X = testCase.TestData.X;
    cpp_rows = capture_error(@() dtwc_mex('tier1_cluster', X, 2, 'pam', -1, ...
        '', 100, 0, -1, '', ''));
    verifyEqual(testCase, cpp_rows.identifier, 'dtwc:invalidArgument');
    verifyEqual(testCase, cpp_rows.message, 'load: skip_rows must be non-negative.');
    cpp_cols = capture_error(@() dtwc_mex('tier1_cluster', X, 2, 'pam', -1, ...
        '', 100, -1, 0, '', ''));
    verifyEqual(testCase, cpp_cols.identifier, 'dtwc:invalidArgument');
    verifyEqual(testCase, cpp_cols.message, 'load: skip_cols must be non-negative.');
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

function test_dtwclustering_gpu_index_is_parsed_from_the_canonical_name(testCase)
%   S3: the resolver behind Device='gpu:N'. Env::set_device parses the suffix
%   into device_index() and reports it back in the canonical name, so the
%   estimator's resolver must read the same ordinal out of that name.
    verifyEqual(testCase, dtwc.DTWClustering.gpu_index('cpu'), 0);
    verifyEqual(testCase, dtwc.DTWClustering.gpu_index('gpu'), 0);
    verifyEqual(testCase, dtwc.DTWClustering.gpu_index('gpu:1'), 1);
    verifyEqual(testCase, dtwc.DTWClustering.gpu_index('cuda:3'), 3);
    verifyError(testCase, @() dtwc.DTWClustering.gpu_index('gpu:x'), ...
        'dtwc:deviceError');
end

function test_dtwclustering_forwards_the_gpu_ordinal_to_cuda_settings(testCase)
%   S3: DTWClustering.fit set the CUDA strategy but never the device id, so
%   'gpu:1' silently executed on GPU 0 -- C++ configure_device (dtwc/api.cpp)
%   sets cuda_settings.device_id = index. Capability-branched rather than
%   assumption-filtered: an Incomplete is a silent skip that the matlab_suite
%   gate rejects, so BOTH builds must assert something here.
    info = dtwc_mex('system_check');
    if info.cuda || info.metal
        prob = dtwc.Problem('gpu_ordinal');
        prob.set_data(testCase.TestData.X);
        dtwc.DTWClustering.apply_device_strategy(prob, 'gpu:1');
        verifyEqual(testCase, prob.get_cuda_settings().device_id, 1);
        fprintf('S3_GPU_ORDINAL branch=gpu observed_device_id=%d\n', ...
                prob.get_cuda_settings().device_id);
    else
        % No GPU backend: Env must reject the request before fit() creates a
        % Problem, and the process device must be left untouched.
        dtwc.device('cpu');
        c = dtwc.DTWClustering('NClusters', 2, 'Device', 'gpu:1');
        verifyError(testCase, @() c.fit(testCase.TestData.X), 'dtwc:deviceError');
        verifyEqual(testCase, dtwc.device(), 'cpu');
        fprintf('S3_GPU_ORDINAL branch=no-gpu rejected-before-effect\n');
    end
end

function test_problem_cuda_settings_round_trip(testCase)
%   §2.1 set_cuda_settings/get_cuda_settings; omitting precision keeps it.
    prob = dtwc.Problem('cuda_roundtrip');
    verifyEqual(testCase, prob.get_cuda_settings(), ...
        struct('device_id', 0, 'precision', 0));
    prob.set_cuda_settings(2, 1);
    verifyEqual(testCase, prob.get_cuda_settings(), ...
        struct('device_id', 2, 'precision', 1));
    prob.set_cuda_settings(3);
    verifyEqual(testCase, prob.get_cuda_settings(), ...
        struct('device_id', 3, 'precision', 1));
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

function test_problem_set_mip_settings_lr_max_nodes(testCase)
%   §2.1 lr_max_nodes round-trips and rejects a non-integer, as other int fields do.
    prob = dtwc.Problem('lr_nodes');
    verifyEqual(testCase, prob.get_mip_settings().lr_max_nodes, 2000000);
    prob.set_mip_settings(struct('lr_max_nodes', 12345));
    verifyEqual(testCase, prob.get_mip_settings().lr_max_nodes, 12345);
    verifyError(testCase, @() prob.set_mip_settings(struct('lr_max_nodes', 1.5)), ...
                ?MException);
    verifyEqual(testCase, prob.get_mip_settings().lr_max_nodes, 12345);
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

% =========================================================================
%  F22 - retained MATLAB aliases warn once and forward
% =========================================================================

function test_f22_matlab_deprecation_policy(testCase)
%   F22: drive every frozen MATLAB compatibility alias through its public
%   +dtwc entry point. Each old operation must issue exactly one stable
%   warning, its canonical twin must remain silent, and results must match.
    import matlab.unittest.constraints.IssuesNoWarnings
    import matlab.unittest.constraints.IssuesWarnings

    warningId = 'dtwc:deprecatedAlias';
    warningState = warning;
    warningCleanup = onCleanup(@() warning(warningState)); %#ok<NASGU>
    warning('on', warningId);

    X = testCase.TestData.X;
    D = [0 2 5 9 14 20; ...
         2 0 4 8 13 19; ...
         5 4 0 3  7 12; ...
         9 8 3 0  2  6; ...
        14 13 7 2 0  3; ...
        20 19 12 6 3 0];
    labelsTrue = int32([1 1 1 2 2 2 3 3]);
    labelsPred = int32([1 1 2 2 2 3 3 3]);

    rows = {
        'dtwc.Problem.Band', ...
            'dtwc.Problem.set_band', ...
            @() f22_config_operation(X, 'Band', 3, false), ...
            @() f22_config_operation(X, 'Band', 3, true);
        'dtwc.Problem.Verbose', ...
            'dtwc.Problem.set_verbose', ...
            @() f22_config_operation(X, 'Verbose', true, false), ...
            @() f22_config_operation(X, 'Verbose', true, true);
        'dtwc.Problem.MaxIter', ...
            'dtwc.Problem.set_max_iter', ...
            @() f22_config_operation(X, 'MaxIter', 7, false), ...
            @() f22_config_operation(X, 'MaxIter', 7, true);
        'dtwc.Problem.NRepetition', ...
            'dtwc.Problem.set_n_repetitions', ...
            @() f22_config_operation(X, 'NRepetition', 3, false), ...
            @() f22_config_operation(X, 'NRepetition', 3, true);
        'dtwc.Problem.get_distance_matrix', ...
            'dtwc.Problem.distance_matrix', ...
            @() f22_distance_read_operation(X, D, false), ...
            @() f22_distance_read_operation(X, D, true);
        'dtwc.Problem.Size', ...
            'dtwc.Problem.size', ...
            @() f22_read_property_operation(X, 'Size', false), ...
            @() f22_read_property_operation(X, 'Size', true);
        'dtwc.Problem.ClusterSize', ...
            'dtwc.Problem.n_clusters', ...
            @() f22_read_property_operation(X, 'ClusterSize', false), ...
            @() f22_read_property_operation(X, 'ClusterSize', true);
        'dtwc.Problem.Name', ...
            'dtwc.Problem.name', ...
            @() f22_read_property_operation(X, 'Name', false), ...
            @() f22_read_property_operation(X, 'Name', true);
        'dtwc.Problem.CentroidsInd', ...
            'dtwc.Problem.medoids', ...
            @() f22_read_property_operation(X, 'CentroidsInd', false), ...
            @() f22_read_property_operation(X, 'CentroidsInd', true);
        'dtwc.Problem.ClustersInd', ...
            'dtwc.Problem.labels', ...
            @() f22_read_property_operation(X, 'ClustersInd', false), ...
            @() f22_read_property_operation(X, 'ClustersInd', true);
        'dtwc.davies_bouldin_index', ...
            'dtwc.davies_bouldin', ...
            @() f22_problem_score_operation(X, 'davies_bouldin_index'), ...
            @() f22_problem_score_operation(X, 'davies_bouldin');
        'dtwc.dunn_index', ...
            'dtwc.dunn', ...
            @() f22_problem_score_operation(X, 'dunn_index'), ...
            @() f22_problem_score_operation(X, 'dunn');
        'dtwc.calinski_harabasz_index', ...
            'dtwc.calinski_harabasz', ...
            @() f22_problem_score_operation(X, 'calinski_harabasz_index'), ...
            @() f22_problem_score_operation(X, 'calinski_harabasz');
        'dtwc.adjusted_rand_index', ...
            'dtwc.adjusted_rand', ...
            @() f22_label_score_operation( ...
                labelsTrue, labelsPred, 'adjusted_rand_index'), ...
            @() f22_label_score_operation( ...
                labelsTrue, labelsPred, 'adjusted_rand');
        'dtwc.normalized_mutual_information', ...
            'dtwc.normalized_mutual_info', ...
            @() f22_label_score_operation( ...
                labelsTrue, labelsPred, 'normalized_mutual_information'), ...
            @() f22_label_score_operation( ...
                labelsTrue, labelsPred, 'normalized_mutual_info')
    };

    verifyEqual(testCase, size(rows, 1), 15, ...
        'F22 frozen MATLAB alias inventory changed.');

    warningProfiles = 0;
    messages = 0;
    canonicalSilent = 0;
    equivalent = 0;

    for i = 1:size(rows, 1)
        oldName = rows{i, 1};
        newName = rows{i, 2};
        aliasOperation = rows{i, 3};
        canonicalOperation = rows{i, 4};
        expectedMessage = sprintf( ...
            '''%s'' is deprecated; use ''%s'' instead.', oldName, newName);

        warningConstraint = IssuesWarnings( ...
            {warningId}, 'Exactly', true, 'WhenNargoutIs', 3);
        warningProfileOk = warningConstraint.satisfiedBy( ...
            @() f22_invoke_and_capture_warning(aliasOperation));
        warningOutputs = warningConstraint.FunctionOutputs;
        aliasValue = warningOutputs{1};
        actualMessage = warningOutputs{2};
        actualId = warningOutputs{3};

        warningProfiles = warningProfiles + double(warningProfileOk);
        messageOk = strcmp(actualId, warningId) && ...
                    strcmp(actualMessage, expectedMessage);
        messages = messages + double(messageOk);
        verifyTrue(testCase, warningProfileOk, sprintf( ...
            'F22 warning ID/count mismatch for %s.', oldName));
        verifyTrue(testCase, messageOk, sprintf( ...
            'F22 warning message mismatch for %s.', oldName));

        noWarningConstraint = IssuesNoWarnings('WhenNargoutIs', 1);
        canonicalIsSilent = noWarningConstraint.satisfiedBy(canonicalOperation);
        canonicalValue = noWarningConstraint.FunctionOutputs{1};
        canonicalSilent = canonicalSilent + double(canonicalIsSilent);
        verifyTrue(testCase, canonicalIsSilent, sprintf( ...
            'F22 canonical operation warned for %s.', newName));

        valuesMatch = isequaln(aliasValue, canonicalValue);
        equivalent = equivalent + double(valuesMatch);
        verifyTrue(testCase, valuesMatch, sprintf( ...
            'F22 behavior mismatch: %s versus %s.', oldName, newName));
        f22_verify_nondegenerate_value(testCase, oldName, aliasValue, D);
    end

    f22_verify_config_setter_atomicity(testCase);
    f22_verify_warning_precedes_config_effect(testCase, warningId);

    constructorConstraint = IssuesNoWarnings('WhenNargoutIs', 1);
    constructorSilent = constructorConstraint.satisfiedBy( ...
        @() dtwc.Problem('f22_constructor_silent'));
    constructed = constructorConstraint.FunctionOutputs{1};
    verifyTrue(testCase, constructorSilent, ...
        'dtwc.Problem construction emitted a deprecation warning.');
    assertClass(testCase, constructed, 'dtwc.Problem');
    assertEqual(testCase, constructed.name(), 'f22_constructor_silent');

    writerConstraint = IssuesNoWarnings('WhenNargoutIs', 1);
    writerSilent = writerConstraint.satisfiedBy( ...
        @() f22_silent_distance_writer(X, D));
    verifyTrue(testCase, writerSilent, ...
        'Canonical Problem.set_distance_matrix emitted a warning.');
    assertEqual(testCase, writerConstraint.FunctionOutputs{1}, D);

    tier1Constraint = IssuesNoWarnings('WhenNargoutIs', 1);
    tier1Silent = tier1Constraint.satisfiedBy(@() f22_silent_tier1_fit(X));
    fitted = tier1Constraint.FunctionOutputs{1};
    verifyTrue(testCase, tier1Silent, ...
        'Canonical DTWClustering.fit emitted a deprecation warning.');
    assertClass(testCase, fitted, 'dtwc.DTWClustering');
    assertNumElements(testCase, fitted.Labels, size(X, 1));
    assertNumElements(testCase, fitted.MedoidIndices, 2);

    allPass = warningProfiles == 15 && messages == 15 && ...
              canonicalSilent == 15 && equivalent == 15 && ...
              constructorSilent && writerSilent && tier1Silent;
    verdict = 'FAIL';
    if allPass
        verdict = 'PASS';
    end
    fprintf(['F22_MATLAB_DEPRECATION aliases=15/15 ' ...
             'warning_profiles=%d/15 messages=%d/15 ' ...
             'canonical_silent=%d/15 equivalence=%d/15 ' ...
             'constructor_silent=%d/1 tier1_silent=%d/1 ' ...
             'skips=0 verdict=%s\n'], ...
            warningProfiles, messages, canonicalSilent, equivalent, ...
            constructorSilent, tier1Silent, verdict);
    verifyTrue(testCase, allPass, ...
        'F22 MATLAB deprecation contract is incomplete.');
end

function err = capture_error(fn)
%CAPTURE_ERROR Run fn and return the MException it must raise.
    err = [];
    try
        fn();
    catch caught
        err = caught;
    end
    assert(~isempty(err), 'expected an error, none was raised');
end

function [value, message, identifier] = f22_invoke_and_capture_warning(operation)
    lastwarn('');
    value = operation();
    [message, identifier] = lastwarn;
end

function value = f22_config_operation(X, propertyName, candidate, canonical)
    prob = dtwc.Problem(['f22_config_' lower(propertyName)]);
    prob.set_data(X);

    if canonical
        switch propertyName
            case 'Band'
                prob.set_band(candidate);
            case 'Verbose'
                prob.set_verbose(candidate);
            case 'MaxIter'
                prob.set_max_iter(candidate);
            case 'NRepetition'
                prob.set_n_repetitions(candidate);
            otherwise
                error('dtwc:f22TestOracle', ...
                    'Unknown canonical configuration property: %s.', propertyName);
        end
    else
        prob.(propertyName) = candidate;
    end

    switch propertyName
        case 'Band'
            info = dtwc_mex('Problem_get_info', prob.get_handle());
            value = struct('cached', prob.Band, 'native', info.band);
        case 'Verbose'
            info = dtwc_mex('Problem_get_info', prob.get_handle());
            value = struct('cached', prob.Verbose, 'native', info.verbose);
        case 'MaxIter'
            info = dtwc_mex('Problem_get_info', prob.get_handle());
            value = struct('cached', prob.MaxIter, ...
                           'native', info.max_iter);
        case 'NRepetition'
            info = dtwc_mex('Problem_get_info', prob.get_handle());
            value = struct('cached', prob.NRepetition, ...
                           'native', info.n_repetitions);
        otherwise
            error('dtwc:f22TestOracle', ...
                'Unknown configuration observation: %s.', propertyName);
    end
end

function value = f22_distance_read_operation(X, D, canonical)
    prob = dtwc.Problem('f22_distance_read');
    prob.set_data(X);
    prob.set_distance_matrix(D);
    if canonical
        value = prob.distance_matrix();
    else
        value = prob.get_distance_matrix();
    end
end

function value = f22_read_property_operation(X, propertyName, canonical)
    prob = f22_clustered_problem(X, 'f22_read_alias');
    if canonical
        switch propertyName
            case 'Size'
                value = prob.size();
            case 'ClusterSize'
                value = prob.n_clusters();
            case 'Name'
                value = prob.name();
            case 'CentroidsInd'
                value = prob.medoids();
            case 'ClustersInd'
                value = prob.labels();
            otherwise
                error('dtwc:f22TestOracle', ...
                    'Unknown canonical read property: %s.', propertyName);
        end
    else
        value = prob.(propertyName);
    end
end

function value = f22_problem_score_operation(X, scoreName)
    prob = f22_clustered_problem(X, 'f22_problem_score');
    scoreFunction = str2func(['dtwc.' scoreName]);
    value = scoreFunction(prob);
end

function value = f22_label_score_operation(labelsTrue, labelsPred, scoreName)
    scoreFunction = str2func(['dtwc.' scoreName]);
    value = scoreFunction(labelsTrue, labelsPred);
end

function prob = f22_clustered_problem(X, name)
    prob = dtwc.Problem(name);
    prob.set_data(X);
    dtwc.fast_pam(prob, 2, 'MaxIter', 20, 'Seed', 42);
end

function D = f22_silent_distance_writer(X, D)
    prob = dtwc.Problem('f22_canonical_writer');
    prob.set_data(X);
    prob.set_distance_matrix(D);
    D = prob.distance_matrix();
end

function fitted = f22_silent_tier1_fit(X)
    estimator = dtwc.DTWClustering( ...
        'NClusters', 2, 'Metric', 'l1', 'Device', 'cpu', 'NInit', 1);
    fitted = estimator.fit(X);
end

function f22_verify_nondegenerate_value(testCase, oldName, value, D)
    switch oldName
        case 'dtwc.Problem.Band'
            assertEqual(testCase, value.cached, 3);
            assertEqual(testCase, value.native, 3);
        case 'dtwc.Problem.Verbose'
            assertTrue(testCase, value.cached);
            assertTrue(testCase, value.native);
        case 'dtwc.Problem.MaxIter'
            assertEqual(testCase, value.cached, 7);
            assertEqual(testCase, value.native, 7);
        case 'dtwc.Problem.NRepetition'
            assertEqual(testCase, value.cached, 3);
            assertEqual(testCase, value.native, 3);
        case 'dtwc.Problem.get_distance_matrix'
            assertEqual(testCase, value, D);
        case 'dtwc.Problem.Size'
            assertEqual(testCase, value, 6);
        case 'dtwc.Problem.ClusterSize'
            assertEqual(testCase, value, 2);
        case 'dtwc.Problem.Name'
            assertEqual(testCase, value, 'f22_read_alias');
        case 'dtwc.Problem.CentroidsInd'
            assertNumElements(testCase, value, 2);
            assertTrue(testCase, all(value >= 1 & value <= 6));
        case 'dtwc.Problem.ClustersInd'
            assertNumElements(testCase, value, 6);
            assertTrue(testCase, all(value >= 1 & value <= 2));
        case {'dtwc.davies_bouldin_index', ...
              'dtwc.dunn_index', ...
              'dtwc.calinski_harabasz_index'}
            assertTrue(testCase, isscalar(value) && isfinite(value));
            assertGreaterThan(testCase, value, 0);
        case {'dtwc.adjusted_rand_index', ...
              'dtwc.normalized_mutual_information'}
            assertTrue(testCase, isscalar(value) && isfinite(value));
            assertGreaterThan(testCase, value, 0);
            assertLessThan(testCase, value, 1);
        otherwise
            error('dtwc:f22TestOracle', ...
                'Unknown F22 non-degenerate observation: %s.', oldName);
    end
end

function f22_verify_config_setter_atomicity(testCase)
%   A rejected canonical value must not update MATLAB's cached observation
%   ahead of the native setter. These asymmetric vector candidates expose
%   any scalar-boundary bug that consumes only the first native element.
    prob = dtwc.Problem('f22_config_atomicity');

    prob.set_band(3);
    assertError(testCase, @() prob.set_band([4 5]), ...
        'dtwc:invalidArgument');
    info = dtwc_mex('Problem_get_info', prob.get_handle());
    assertEqual(testCase, prob.Band, 3);
    assertEqual(testCase, info.band, 3);

    prob.set_verbose(true);
    assertError(testCase, @() prob.set_verbose([false true]), ...
        'dtwc:invalidArgument');
    info = dtwc_mex('Problem_get_info', prob.get_handle());
    assertTrue(testCase, prob.Verbose);
    assertTrue(testCase, info.verbose);

    prob.set_max_iter(7);
    assertError(testCase, @() prob.set_max_iter([8 9]), ...
        'dtwc:invalidArgument');
    info = dtwc_mex('Problem_get_info', prob.get_handle());
    assertEqual(testCase, prob.MaxIter, 7);
    assertEqual(testCase, info.max_iter, 7);

    prob.set_n_repetitions(3);
    assertError(testCase, @() prob.set_n_repetitions([4 5]), ...
        'dtwc:invalidArgument');
    info = dtwc_mex('Problem_get_info', prob.get_handle());
    assertEqual(testCase, prob.NRepetition, 3);
    assertEqual(testCase, info.n_repetitions, 3);
end

function f22_verify_warning_precedes_config_effect(testCase, warningId)
%   Escalating the compatibility warning to an error must stop each
%   mutating alias before either the MATLAB cache or native state changes.
    prob = dtwc.Problem('f22_config_warning_order');
    prob.set_band(3);
    prob.set_verbose(true);
    prob.set_max_iter(7);
    prob.set_n_repetitions(3);

    warningState = warning;
    warningCleanup = onCleanup(@() warning(warningState)); %#ok<NASGU>
    warning('error', warningId);

    assertError(testCase, @() f22_assign_config_alias(prob, 'Band', 4), ...
        warningId);
    assertError(testCase, ...
        @() f22_assign_config_alias(prob, 'Verbose', false), warningId);
    assertError(testCase, @() f22_assign_config_alias(prob, 'MaxIter', 8), ...
        warningId);
    assertError(testCase, ...
        @() f22_assign_config_alias(prob, 'NRepetition', 4), warningId);

    info = dtwc_mex('Problem_get_info', prob.get_handle());
    assertEqual(testCase, prob.Band, 3);
    assertTrue(testCase, prob.Verbose);
    assertEqual(testCase, prob.MaxIter, 7);
    assertEqual(testCase, prob.NRepetition, 3);
    assertEqual(testCase, info.band, 3);
    assertTrue(testCase, info.verbose);
    assertEqual(testCase, info.max_iter, 7);
    assertEqual(testCase, info.n_repetitions, 3);
end

function f22_assign_config_alias(prob, propertyName, value)
    prob.(propertyName) = value;
end
