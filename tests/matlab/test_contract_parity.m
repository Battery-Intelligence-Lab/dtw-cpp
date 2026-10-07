function tests = test_contract_parity
%TEST_CONTRACT_PARITY MATLAB parity gate for the documented Tier-1 and Tier-2 API.
%
%   Asserts that every public name the Tier-1 and Tier-2 pages
%   (docs/content/api/tier-1.md, tier-2.md) document for MATLAB is present and
%   callable through the +dtwc package / dtwc_mex gateway. This is the Phase 2
%   Task 2.2 parity gate: it drives the real public entry points (Tier-1
%   device/load/cluster/Result, Tier-2 Problem setters+methods, scores,
%   algorithms, distance functions, and the checkpoint surface), not dead
%   siblings.
%
%   Each test comment names the public entry point it exercises (LESSONS: tests
%   pin the live code path).
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
%  Tier 1
% =========================================================================

function test_version_matches_ssot(testCase)
%   The MEX reports the repository's VERSION file.
    repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    expected = strtrim(fileread(fullfile(repoRoot, 'VERSION')));
    verifyEqual(testCase, dtwc_mex('version'), expected);
end

function test_diagnostics_return_the_cpp_report_fields(testCase)
%   The MEX copies each C++ report into a struct by hand; what the fields say
%   is tests/unit/test_test_api.cpp's.
    verifyEqual(testCase, ...
        {sort(fieldnames(dtwc.test.parallelisation())'), sort(fieldnames(dtwc.test.gpu())')}, ...
        {sort({'available', 'max_threads', 'threads_engaged', 'pass', 'reason'}), ...
         sort({'available', 'backend', 'device_name', 'validated', 'pass', 'reason'})});
end

function test_tier1_device_get_set(testCase)
%   dtwc.device -> MEX set_device/get_device -> dtwc::device().
    name = dtwc.device('cpu');
    verifyEqual(testCase, name, 'cpu');
    verifyEqual(testCase, dtwc.device(), 'cpu');
end

function test_tier1_device_unknown_raises_deviceError(testCase)
%   No silent fallback: an unknown device name -> dtwc:deviceError.
    verifyError(testCase, @() dtwc.device('definitely_not_a_device'), 'dtwc:deviceError');
end

function test_tier1_load_matrix_and_options(testCase)
%   dtwc.load -> dtwc.Dataset (lazy handle, no I/O for a matrix source).
    ds = dtwc.load(testCase.TestData.X, 'SkipCols', 0, 'Delimiter', '', 'Name', 'ptest');
    verifyClass(testCase, ds, 'dtwc.Dataset');
    verifyEqual(testCase, ds.Name, 'ptest');
end

function test_tier1_load_skip_rows(testCase)
%   skip_rows: header LINES for a path, leading SERIES for a matrix.
    ds = dtwc.load(testCase.TestData.X, 'SkipRows', 2);
    verifyEqual(testCase, ds.SkipRows, 2);

    % The path is read by C++ inside dtwc.cluster: two header lines and the id
    % column are dropped, leaving the four series below (cost 0.2 pins values).
    f = [tempname '.csv'];
    fid = fopen(f, 'w');
    fprintf(fid, 'id,t0,t1\nunit,s,s\na,0,0\nb,0.1,0\nc,10,11\nd,10,10.9\n');
    fclose(fid);
    c = onCleanup(@() delete(f));
    hdr = dtwc.load(f, 'SkipCols', 1, 'SkipRows', 2, 'Delimiter', ',');
    fromFile = dtwc.cluster(hdr, 2);
    fromMemory = dtwc.cluster([0 0; 0.1 0; 10 11; 10 10.9], 2);
    verifyEqual(testCase, fromFile.labels, fromMemory.labels);
    verifyEqual(testCase, fromFile.cost, fromMemory.cost, 'AbsTol', 1e-12);
    verifyEqual(testCase, fromFile.cost, 0.2, 'AbsTol', 1e-12);
end

function test_tier1_load_negative_skip_rows_rejected(testCase)
%   SkipRows is checked exactly as SkipCols is, where the handle is made,
%   before any file is read; the MEX reader keeps its own guard (C++'s words).
    for key = {'SkipCols', 'SkipRows'}
        err = capture_error(@() dtwc.load(testCase.TestData.X, key{1}, -1));
        verifyEqual(testCase, err.identifier, 'dtwc:invalidArgument');
        verifyEqual(testCase, err.message, sprintf('load: %s must be a non-negative integer.', key{1}));
    end
    err = capture_error(@() dtwc_mex('read_data', 'unread.csv', 0, -1, ''));
    verifyEqual(testCase, err.identifier, 'dtwc:invalidArgument');
    verifyEqual(testCase, err.message, 'load: skip_rows must be non-negative.');
end

function test_tier1_cluster_returns_result(testCase)
%   dtwc.cluster -> dtwc.Result (Tier-1 pam path).
    res = dtwc.cluster(testCase.TestData.X, testCase.TestData.k, ...
                       'Method', 'pam', 'Band', -1, 'MaxIter', 50);
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

    tier1 = dtwc.cluster(X, 3, 'Method', 'pam');

    prob42 = dtwc.Problem('seed42');
    prob42.set_data(X);
    seeded42 = dtwc.fast_pam(prob42, 3, 'Seed', 42);

    prob29 = dtwc.Problem('seed29');
    prob29.set_data(X);
    seeded29 = dtwc.fast_pam(prob29, 3, 'Seed', 29);

    verifyEqual(testCase, seeded42.medoid_indices, [7 3 6]);
    verifyEqual(testCase, seeded42.labels, [2 2 2 2 3 3 1 1]);
    verifyEqual(testCase, seeded42.total_cost, 24);
    verifyEqual(testCase, seeded29.medoid_indices, [5 2 8]);
    verifyEqual(testCase, seeded29.labels, [2 2 2 1 1 1 3 3]);
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
    verifyEqual(testCase, one.Inertia, 24);
    verifyEqual(testCase, two.Inertia, 20);
    verifyLessThan(testCase, two.Inertia, one.Inertia);
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
    routed_l1 = routed_matrix(X, 'l1');
    routed_squared = routed_matrix(X, 'squared_euclidean');
    assertEqual(testCase, routed_l1, D_l1);
    assertEqual(testCase, routed_squared, D_squared);

    l1_problem = dtwc.Problem('F18_l1_oracle');
    l1_problem.set_data(X);
    l1_problem.set_distance_matrix(D_l1);
    l1_result = dtwc.fast_pam(l1_problem, 2, 'MaxIter', 100, 'Seed', 42);
    % Seed 42 ends in the swap-local optimum {4, 1}: no single swap of it is
    % cheaper than 10. The global optimum {3, 2}, cost 9, is what seed 43 finds,
    % and what the NInit = 2 estimator below reports.
    assertEqual(testCase, l1_result.labels, [2 1 1 1]);
    assertEqual(testCase, l1_result.medoid_indices, [4 1]);
    assertEqual(testCase, l1_result.total_cost, 10);
    assertEqual(testCase, l1_result.iterations, 1);
    assertTrue(testCase, l1_result.converged);

    squared_problem = dtwc.Problem('F18_squared_oracle');
    squared_problem.set_data(X);
    squared_problem.set_distance_matrix(D_squared);
    squared_result = dtwc.fast_pam(squared_problem, 2, ...
                                   'MaxIter', 100, 'Seed', 42);
    assertEqual(testCase, squared_result.labels, [2 1 1 1]);
    assertEqual(testCase, squared_result.medoid_indices, [4 1]);
    assertEqual(testCase, squared_result.total_cost, 30);
    assertEqual(testCase, squared_result.iterations, 1);
    assertTrue(testCase, squared_result.converged);

    l1_estimator = dtwc.DTWClustering( ...
        'NClusters', 2, 'Metric', 'l1', 'Device', 'cpu', 'NInit', 2);
    l1_estimator = l1_estimator.fit(X);
    assertEqual(testCase, l1_estimator.Labels, [1 2 1 1]);
    assertEqual(testCase, l1_estimator.MedoidIndices, [3 2]);
    assertEqual(testCase, l1_estimator.Inertia, 9);

    squared_estimator = dtwc.DTWClustering( ...
        'NClusters', 2, 'Metric', 'squared_euclidean', ...
        'Device', 'cpu', 'NInit', 2);
    squared_estimator = squared_estimator.fit(X);
    assertEqual(testCase, squared_estimator.Labels, [2 1 1 1]);
    assertEqual(testCase, squared_estimator.MedoidIndices, [4 1]);
    assertEqual(testCase, squared_estimator.Inertia, 30);

    uppercase_estimator = dtwc.DTWClustering( ...
        'NClusters', 2, 'Metric', 'SQUARED_EUCLIDEAN', ...
        'Device', 'cpu', 'NInit', 2);
    uppercase_labels = uppercase_estimator.fit_predict(X);
    assertEqual(testCase, uppercase_labels, [2 1 1 1]);

    assertNotEqual(testCase, l1_estimator.Labels, squared_estimator.Labels);
    assertNotEqual(testCase, l1_estimator.MedoidIndices, ...
                   squared_estimator.MedoidIndices);
    assertNotEqual(testCase, l1_estimator.Inertia, ...
                   squared_estimator.Inertia);

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
        'unknown metric ''not_a_metric''. Valid: l1, squared_euclidean.');
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
        'NClusters', 2, 'Variant', 'ddtw', 'Device', 'cpu', 'NInit', 2);
    bad_missing.MissingStrategy = 'zero_cost';
    try
        bad_missing.fit(X);
    catch caught
        missing_error = caught;
    end
    assertFalse(testCase, isempty(missing_error), ...
                'DDTW plus zero_cost must fail loudly.');
    assertEqual(testCase, missing_error.identifier, 'dtwc:invalidArgument');

    % MaxIter = 0 would report the initial medoids' cost as the clustering's.
    zero_error = capture_error(@() dtwc.DTWClustering('NClusters', 2, 'MaxIter', 0).fit(X));
    assertEqual(testCase, zero_error.identifier, 'dtwc:invalidArgument');
    verifySubstring(testCase, zero_error.message, 'max_iter must be at least 1');

    fprintf(['F18_MATLAB_VALIDATION unknown_metric=1/1 ' ...
        'unknown_precedence=1/1 squared_variant=1/1 ' ...
        'variant_missing=1/1 skips=0\n']);
end

function test_dtwclustering_computes_every_metric_cpp_computes(testCase)
%   DDTW and the missing-data strategies take a metric in C++, so the fitted
%   cost is the sum of each series' dtwc.distance.dtw to its medoid.
    cases = {
        {'Variant', 'ddtw', 'Metric', 'squared_euclidean'}, ...
            [0 1 3 6; 0 1 2 4; 5 5 6 6; 5 6 6 7]
        {'MissingStrategy', 'zero_cost', 'Metric', 'squared_euclidean'}, ...
            [0 NaN 1 0; 0 1 1 0; 9 9 NaN 8; 9 8 8 8]
    };
    for i = 1:size(cases, 1)
        [settings, X] = cases{i, :};
        c = dtwc.DTWClustering('NClusters', 2, 'Device', 'cpu', settings{:}).fit(X);
        cost = 0;
        for s = 1:size(X, 1)
            cost = cost + dtwc.distance.dtw(X(s, :), X(c.MedoidIndices(c.Labels(s)), :), ...
                                            settings{:});
        end
        verifyEqual(testCase, c.Inertia, cost, 'RelTol', 1e-12, sprintf('case %d', i));
    end
end

function test_dtwclustering_predict_and_score_read_the_fitted_medoids(testCase)
%   predict is the nearest medoid under the fitted distance and score minus the
%   total distance to the nearest medoids, both by dtwc.distance.dtw: on the
%   training series they are C++'s Labels and -Inertia, and nothing is refitted.
    X = [0 1; 3 8; 5 2; 6 4];
    c = dtwc.DTWClustering('NClusters', 2, 'NInit', 2).fit(X);
    verifyEqual(testCase, c.predict(X), c.Labels);
    verifyEqual(testCase, c.score(X), -c.Inertia, 'AbsTol', 1e-12);
    verifyEqual(testCase, c.predict({[6 4 4], 0}), [1 1]);
    verifyError(testCase, @() c.predict({magic(3)}), 'dtwc:invalidArgument');   % as fit refuses it
    verifyEqual(testCase, [c.Labels; c.MedoidIndices(c.Labels)], [1 2 1 1; 3 2 3 3]);
end

function test_keys_are_python_words_in_camel_case(testCase)
%   Volkan 10-01: the same words in every language, each in its own case.
%   dtwc.cluster's keys are Python's cluster() keywords (the binding's Config
%   less n_clusters, which k names) and DTWClustering's settable properties are
%   Python's DTWClustering parameters, in CamelCase. The words are those of
%   python/src/_dtwcpp_core.cpp's Config and python/dtwcpp/_clustering.py.
    cluster_words = {'name', 'method', 'band', 'metric', 'variant', 'max_iter', 'n_init', ...
        'dc', 'wdtw_g', 'adtw_penalty', 'sdtw_gamma', 'msm_c', 'twe_nu', 'twe_lambda', ...
        'mv_mode', 'missing_strategy', 'sample_size', 'n_samples', 'seed', 'batch_size', ...
        'linkage', 'solver', 'mip_gap', 'time_limit', 'no_warm_start', 'numeric_focus', ...
        'mip_focus', 'verbose_solver', 'lr_max_nodes', 'device', 'gpu_precision', 'verbose'};
    estimator_words = {'n_clusters', 'method', 'variant', 'band', 'max_iter', 'n_init', ...
        'wdtw_g', 'adtw_penalty', 'msm_c', 'twe_nu', 'twe_lambda', 'mv_mode', ...
        'missing_strategy', 'metric', 'batch_size', 'random_state', 'device'};
    camel = @(words) sort(cellfun(@(w) strjoin(cellfun(@(p) [upper(p(1)) p(2:end)], ...
        strsplit(w, '_'), 'UniformOutput', false), ''), words, 'UniformOutput', false));

    err = capture_error(@() dtwc.cluster(testCase.TestData.X, 2, 'NoSuchKey', 1));
    verifyEqual(testCase, err.identifier, 'dtwc:invalidArgument');
    listed = regexp(err.message, '^unknown key ''NoSuchKey''\. Valid: (.*)\.$', 'tokens', 'once');
    verifyEqual(testCase, sort(strsplit(listed{1}, ', ')), camel(cluster_words));

    mc = ?dtwc.DTWClustering;
    settable = {mc.PropertyList(strcmp({mc.PropertyList.SetAccess}, 'public')).Name};
    verifyEqual(testCase, sort(settable), camel(estimator_words));
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
%   Unknown method -> dtwc:invalidArgument (never silently PAM).
    verifyError(testCase, ...
        @() dtwc.cluster(testCase.TestData.X, 2, 'Method', 'no_such_method'), ...
        'dtwc:invalidArgument');
end

function test_tier1_result_score_names(testCase)
%   Result.score(name) for every accepted score name.
    res = dtwc.cluster(testCase.TestData.X, testCase.TestData.k);
    for nm = {'silhouette', 'davies_bouldin', 'dunn', 'calinski_harabasz', 'inertia'}
        s = res.score(nm{1});
        verifyTrue(testCase, isscalar(s) && isnumeric(s), ...
            sprintf('score(%s) must return a numeric scalar', nm{1}));
    end
    verifyError(testCase, @() res.score('nope'), 'dtwc:invalidArgument');
end

function test_tier1_result_save_and_plot(testCase)
%   Result.save(dir) writes the 4 CSVs; Result.plot() renders (headless).
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
%   DTWClustering gains a Device parameter (delegates to dtwc::device()).
    c = dtwc.DTWClustering('NClusters', 2, 'Device', 'cpu');
    verifyEqual(testCase, c.Device, 'cpu');
    c = c.fit(testCase.TestData.X);
    verifyNumElements(testCase, c.Labels, 6);
end

function test_device_names_are_read_by_the_cpp_grammar(testCase)
%   FX-17: MATLAB has no device grammar of its own. dtwc.device and
%   Problem.set_device read a name with C++ dtwc::detail::parse_device (it
%   trims and ignores case) and report it as C++ dtwc::device() does, so a
%   malformed ordinal is C++'s DeviceError, verbatim, on every build.
%   DTWClustering's str2double ordinal parser (gpu_index) is gone.
    previous = dtwc.device();
    restore = onCleanup(@() dtwc.device(previous)); %#ok<NASGU>
    verifyEqual(testCase, dtwc.device(' CPU '), 'cpu');
    unknown = ['[dtwc] unknown device ''gpu:x''. Valid devices: cpu, gpu, ' ...
               'gpu:N (aliases cuda, cuda:N).'];
    prob = dtwc.Problem('grammar');
    for request = {@() dtwc.device('gpu:x'), @() prob.set_device('gpu:x')}
        verifyError(testCase, request{1}, 'dtwc:deviceError');
        try
            request{1}();
        catch err
            verifyEqual(testCase, err.message, unknown);
        end
    end
    mc = meta.class.fromName('dtwc.DTWClustering');
    verifyFalse(testCase, any(strcmp({mc.MethodList.Name}, 'gpu_index')));
end

function test_hpc_is_a_device_error_as_in_cpp(testCase)
%   hpc is Python's: it submits a whole run to SLURM. C++ refuses it with a
%   DeviceError naming Python and slurm_remote.sh; MATLAB reads the same grammar,
%   so every route that takes a device name refuses it alike and the process
%   device is left alone.
    previous = dtwc.device();
    restore = onCleanup(@() dtwc.device(previous)); %#ok<NASGU>
    X = testCase.TestData.X;
    routes = {@() dtwc.device('hpc'), ...
              @() dtwc.Problem('hpc', 'Device', 'hpc'), ...
              @() dtwc.Problem('hpc').set_device('hpc'), ...
              @() dtwc.DTWClustering('NClusters', 2, 'Device', 'hpc').fit(X)};
    for route = routes
        verifyError(testCase, route{1}, 'dtwc:deviceError');
        try
            route{1}();
        catch err
            verifySubstring(testCase, err.message, 'slurm_remote.sh');
        end
    end
    verifyEqual(testCase, dtwc.device(), previous);
end

% =========================================================================
%  Tier 2 — Problem config setters
% =========================================================================

function test_problem_setters_all_callable(testCase)
%   Every MATLAB-column setter on dtwc.Problem.
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
    prob.set_distance('Variant', 'ddtw', 'Metric', 'squared_euclidean');
    ok = prob.set_solver('highs');
    verifyTrue(testCase, islogical(ok));
    prob.set_output_folder(tempdir);
    prob.set_verbose(false);
    prob.set_gpu_precision('auto');
    verifyEqual(testCase, prob.size(), 6);
end

function test_problem_set_mip_settings_roundtrip(testCase)
%   set_mip_settings(struct) + get_mip_settings (MIPSettings).
    prob = dtwc.Problem('mip');
    s = struct('mip_gap', 1e-4, 'time_limit_sec', 30, 'warm_start', false);
    prob.set_mip_settings(s);
    got = prob.get_mip_settings();
    verifyEqual(testCase, got.mip_gap, 1e-4, 'AbsTol', 1e-12);
    verifyEqual(testCase, got.time_limit_sec, 30);
    verifyEqual(testCase, got.warm_start, false);
end

function test_problem_set_mip_settings_lr_max_nodes(testCase)
%   lr_max_nodes round-trips and rejects a non-integer, as other int fields do.
    prob = dtwc.Problem('lr_nodes');
    verifyEqual(testCase, prob.get_mip_settings().lr_max_nodes, 2000000);
    prob.set_mip_settings(struct('lr_max_nodes', 12345));
    verifyEqual(testCase, prob.get_mip_settings().lr_max_nodes, 12345);
    verifyError(testCase, @() prob.set_mip_settings(struct('lr_max_nodes', 1.5)), ...
                ?MException);
    verifyEqual(testCase, prob.get_mip_settings().lr_max_nodes, 12345);
end

function test_problem_set_data_ragged(testCase)
%   Ragged input via a cell array of numeric vectors.
    prob = dtwc.Problem('ragged');
    C = {[1 2 3 4], [1 2 3 4 5 6], [9 9 9]};
    prob.set_data(C);
    verifyEqual(testCase, prob.size(), 3);
end

function test_problem_set_data_names(testCase)
%   Series names alongside the data matrix.
    prob = dtwc.Problem('named');
    names = {'a', 'b', 'c', 'd', 'e', 'f'};
    prob.set_data(testCase.TestData.X, names);
    verifyEqual(testCase, prob.size(), 6);
end

function test_problem_set_data_ndim_multivariate(testCase)
%   ndim multivariate (interleaved layout); L must be divisible by ndim.
    prob = dtwc.Problem('mv');
    prob.set_data(testCase.TestData.X, {}, 2);   % 8 cols -> 4 timesteps x 2 features
    verifyEqual(testCase, prob.size(), 6);
    prob.fill_distance_matrix();
    verifyTrue(testCase, prob.is_distance_matrix_filled());
end

% =========================================================================
%  Tier 2 — Problem distance-matrix & clustering methods
% =========================================================================

function test_problem_methods_all_callable(testCase)
%   refresh/read/max_distance/dist_by_ind/fill/distance_matrix/set/cluster.
    prob = dtwc.Problem('methods');
    prob.set_data(testCase.TestData.X);
    prob.fill_distance_matrix();
    verifyTrue(testCase, prob.is_distance_matrix_filled());

    D = prob.distance_matrix();
    verifySize(testCase, D, [6 6]);
    verifyEqual(testCase, prob.dist_by_ind(1, 2), D(1, 2), 'AbsTol', 1e-10);
    verifyGreaterThanOrEqual(testCase, prob.max_distance(), 0);

    % In-place clustering on the filled matrix. cluster() (Kmedoids/Lloyd) writes
    % per-rep medoid CSVs, so point the output folder at a writable temp dir.
    prob.set_output_folder(tempdir);
    prob.set_n_clusters(2);
    prob.cluster();
    verifyGreaterThanOrEqual(testCase, prob.find_total_cost(), 0);

    % Remaining methods exercised as isolated callable live paths.
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
    prob.set_gpu_precision('fp64');
    verifyFalse(testCase, prob.is_distance_matrix_filled());
end

function test_problem_read_accessors(testCase)
%   Read accessors: size / n_clusters / name / labels / medoids.
    prob = make_filled_problem(testCase);
    verifyEqual(testCase, prob.size(), 6);
    verifyEqual(testCase, prob.n_clusters(), 2);
    verifyEqual(testCase, prob.name(), 'parity');
    verifyNumElements(testCase, prob.labels(), 6);
    verifyNumElements(testCase, prob.medoids(), 2);
end

% =========================================================================
%  Tier 2 — scores (canonical snake_case names)
% =========================================================================

function test_scores_canonical_names(testCase)
%   silhouette/davies_bouldin/dunn/inertia/calinski_harabasz/adjusted_rand/
%   normalized_mutual_info.
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
%  Tier 2 — algorithm free functions
% =========================================================================

function test_algorithms_all_callable(testCase)
%   fast_pam / fast_clara / build_dendrogram / cut_dendrogram.
    prob = dtwc.Problem('algos');
    prob.set_data(testCase.TestData.X);
    prob.fill_distance_matrix();

    r1 = dtwc.fast_pam(prob, 2);
    verifyNumElements(testCase, r1.labels, 6);
    r2 = dtwc.fast_clara(prob, 2, 'NSamples', 2, 'Seed', 42);
    verifyNumElements(testCase, r2.labels, 6);

    dend = dtwc.build_dendrogram(prob, 'Linkage', 'average');
    verifyTrue(testCase, isstruct(dend) && isfield(dend, 'merges'));
    r4 = dtwc.cut_dendrogram(dend, prob, 2);
    verifyNumElements(testCase, r4.labels, 6);
end

% =========================================================================
%  Tier 2 — checkpoint / resume
% =========================================================================

function test_checkpoint_dir_roundtrip(testCase)
%   CheckpointOptions / save_checkpoint / load_checkpoint.
    opts = dtwc.CheckpointOptions('directory', tempdir, 'enabled', true);
    verifyTrue(testCase, isstruct(opts) && islogical(opts.enabled));

    prob = dtwc.Problem('ckpt');
    prob.set_data(testCase.TestData.X);
    prob.fill_distance_matrix();
    ckdir = fullfile(tempdir, ['dtwc_parity_ck_' num2str(feature('getpid'))]);
    dtwc.save_checkpoint(prob, ckdir);

    % The checkpoint file is <dirpath>/<name>.dtwm, named after the Problem.
    prob2 = dtwc.Problem('ckpt');
    prob2.set_data(testCase.TestData.X);
    ok = dtwc.load_checkpoint(prob2, ckdir);
    verifyTrue(testCase, islogical(ok) && ok);
end

% =========================================================================
%  Helpers
% =========================================================================

function D = routed_matrix(X, metric)
%ROUTED_MATRIX The matrix DTWClustering's Problem fills under `metric`.
    prob = dtwc.Problem('F18_routed');
    prob.set_data(X);
    prob.set_distance('Metric', metric);
    prob.fill_distance_matrix();
    D = prob.distance_matrix();
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
