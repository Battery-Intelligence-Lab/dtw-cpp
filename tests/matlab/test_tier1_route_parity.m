function tests = test_tier1_route_parity
%TEST_TIER1_ROUTE_PARITY MATLAB Tier-1 must delegate every decision to C++.
%
%   The MATLAB Tier-1 layer used to re-implement dtwc::cluster()'s routing in
%   .m and drifted from it. These tests pin the CasADi rule: the same names, signatures
%   AND behaviour in C++, Python and MATLAB, with C++ as the reference.
%
%   Each test names the drift item it closes. Oracles are the Tier-2 entry
%   points Problem::cluster() itself calls, so a MATLAB-side re-implementation
%   cannot satisfy them by accident.
%
%   Run with: results = runtests('test_tier1_route_parity');
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
            'SKIPPING dtwc Tier-1 route-parity tests.\n' ...
            'Reason: compiled gateway ''dtwc_mex'' not found on the MATLAB path.\n' bar]);
    end
    rng(42);
    % Two obvious clusters (rows = series); k = 2 is unambiguous.
    testCase.TestData.X = [ones(3, 8); 10 * ones(3, 8)] + 0.01 * randn(6, 8);
    testCase.TestData.k = 2;
    % A larger, less separable set so CLARA's subsampling actually iterates.
    testCase.TestData.Xbig = randn(30, 12) + kron((1:3)', ones(10, 12));
    if testCase.TestData.mex_available
        dtwc.device('cpu');
    end
end

function setup(testCase)
    assumeTrue(testCase, testCase.TestData.mex_available, ...
        'dtwc_mex MEX unavailable - skipping (see loud warning above).');
end

function prob = problem_with(X, name)
    prob = dtwc.Problem(name);
    prob.set_data(X);
end

% =========================================================================
%  Drift 1/2/7: method routing lives in C++, and every C++ name is reachable
% =========================================================================

function test_every_cpp_method_name_is_reachable(testCase)
%   Drift 2: 'onebatch', 'lrcore' and 'tadpole' are valid in C++/Python but
%   raised dtwc:invalidArgument in MATLAB.
    X = testCase.TestData.X;
    k = testCase.TestData.k;
    cwdCleanup = scratch_working_directory(); %#ok<NASGU>
    names = {'pam', 'kmedoids', 'clara', 'onebatch', 'lrcore', 'tadpole', ...
             'hierarchical', 'hclust', 'auto'};
    % 'mip' is the tenth C++ name; it needs an exact MIP backend compiled in,
    % so it is capability-gated -- but this build has HiGHS ON, so the
    % assumption below must HOLD and the printed count must read 10/10.
    skipped = 0;
    if mip_backend_available()
        names{end + 1} = 'mip';
    else
        skipped = 1;
    end
    for i = 1:numel(names)
        res = dtwc.cluster(X, k, 'Method', names{i});
        verifyClass(testCase, res, 'dtwc.Result', names{i});
        verifyNumElements(testCase, res.medoids, k, names{i});
        verifyNumElements(testCase, res.labels, size(X, 1), names{i});
    end
    fprintf('TIER1_METHODS routed=%d/%d skips=%d\n', ...
            numel(names), numel(names) + skipped, skipped);
    assumeTrue(testCase, skipped == 0, ...
        'no exact MIP backend in this MEX - ''mip'' was not routed');
end

% =========================================================================
%  S4: ragged in-memory sources (cell of series), as C++ and Python accept
% =========================================================================

function test_ragged_cell_source_matches_the_tier2_route(testCase)
%   Contract 1.2/1.3: MATLAB Tier-1 must take the same ragged in-memory source
%   C++ load(series_type) and the Python list route take. The oracle is the
%   Tier-2 Problem + seeded FastPAM that Problem::cluster() itself calls (mirrors
%   tests/python/test_api.py::TestRaggedInMemorySource).
    ragged = {[0 0.1 0.2 0.3], [0.05 0.15], [9 9.1 9.2], [9.2 9.05 9.1 9.3 9.15]};
    res = dtwc.cluster(ragged, 2, 'Method', 'pam');
    verifyNumElements(testCase, res.labels, numel(ragged));

    oracle_prob = dtwc.Problem('dataset');
    oracle_prob.set_data(ragged);
    oracle = dtwc.fast_pam(oracle_prob, 2, 'MaxIter', 100, ...
                           'Seed', dtwc.default_random_seed());
    verifyEqual(testCase, res.labels, oracle.labels);
    verifyEqual(testCase, res.medoids, oracle.medoid_indices);
    verifyEqual(testCase, res.cost, oracle.total_cost, 'AbsTol', 1e-12);
end

function test_ragged_cell_source_honours_skip_rows_and_skip_cols(testCase)
%   In memory, SkipRows drops leading SERIES and SkipCols the leading values of
%   every series, as C++ load(series_type) and Python do, for a cell as for a matrix.
    ragged = {[7 7], [1 0 1], [2 5]};
    res = dtwc.cluster(dtwc.load(ragged, 'SkipRows', 1, 'SkipCols', 1), 2);
    trimmed = dtwc.cluster({[0 1], 5}, 2);
    verifyEqual(testCase, res.labels, trimmed.labels);
    verifyEqual(testCase, res.cost, trimmed.cost, 'AbsTol', 1e-12);

    err = capture_error(@() dtwc.cluster(dtwc.load({[0 1 2], 3}, 'SkipCols', 2), 2));
    verifyEqual(testCase, err.identifier, 'dtwc:invalidArgument');
    verifyEqual(testCase, err.message, ...
        'load: SkipCols exceeds an in-memory series length.');
end

function test_ragged_cell_rejects_a_non_numeric_element(testCase)
    err = capture_error(@() dtwc.cluster({[1 2 3], 'oops'}, 2));
    verifyEqual(testCase, err.identifier, 'dtwc:invalidArgument');
    verifyEqual(testCase, err.message, ['data{2} must be of class ''double'' ' ...
        '(got ''char''); convert with double(...) in MATLAB.']);
    % A matrix is not one series, with SkipRows or without: never flattened.
    for skip = 0:1
        err = capture_error(@() dtwc.cluster(dtwc.load({1:3, magic(3), 1:4}, 'SkipRows', skip), 1));
        verifyEqual(testCase, err.message, sprintf('data{%d} must be a vector: one series.', 2 - skip));
    end
end

function test_kmedoids_routes_to_problem_cluster_not_fast_pam(testCase)
%   Drift 1: 'kmedoids' collapsed into the FastPAM call. C++ routes it to
%   Problem::cluster() with Method::Kmedoids -- the oracle below.
    X = testCase.TestData.X;
    k = testCase.TestData.k;
    cwdCleanup = scratch_working_directory(); %#ok<NASGU>
    oracle = problem_with(X, 'dataset');
    oracle.set_band(-1);
    oracle.set_max_iter(100);
    oracle.set_n_clusters(k);
    oracle.set_method('kmedoids');
    oracle.cluster();

    res = dtwc.cluster(X, k, 'Method', 'kmedoids');
    verifyEqual(testCase, res.labels, oracle.labels());
    verifyEqual(testCase, sort(double(res.medoids)), sort(double(oracle.medoids())));
    verifyEqual(testCase, res.cost, oracle.find_total_cost(), 'AbsTol', 1e-12);
end

function test_pam_and_auto_match_seeded_fast_pam(testCase)
%   C++ routes 'pam' (and 'auto' below the CLARA threshold) to the seeded
%   FasterPAM with settings::DEFAULT_RANDOM_SEED.
    X = testCase.TestData.X;
    k = testCase.TestData.k;
    oracle = dtwc.fast_pam(problem_with(X, 'dataset'), k, ...
        'MaxIter', 100, 'Seed', dtwc.default_random_seed());
    for m = {'pam', 'auto'}
        res = dtwc.cluster(X, k, 'Method', m{1});
        verifyEqual(testCase, res.labels, oracle.labels, m{1});
        verifyEqual(testCase, res.medoids, oracle.medoid_indices, m{1});
        verifyEqual(testCase, res.cost, oracle.total_cost, 'AbsTol', 1e-12);
    end
end

function test_hclust_is_an_alias_of_hierarchical(testCase)
    a = dtwc.cluster(testCase.TestData.X, testCase.TestData.k, 'Method', 'hclust');
    b = dtwc.cluster(testCase.TestData.X, testCase.TestData.k, 'Method', 'hierarchical');
    verifyEqual(testCase, a.labels, b.labels);
end

% =========================================================================
%  Drift 4: max_iter must reach the CLARA branch
% =========================================================================

function test_clara_honours_max_iter(testCase)
%   Drift 4: MATLAB forwarded only 'Seed', so fast_clara's own default of 100
%   always won. The oracle is the same algorithm Problem::cluster() calls.
    X = testCase.TestData.Xbig;
    k = 3;
    for mi = [1 100]
        oracle = dtwc.fast_clara(problem_with(X, 'dataset'), k, ...
            'MaxIter', mi, 'Seed', dtwc.default_random_seed());
        res = dtwc.cluster(X, k, 'Method', 'clara', 'MaxIter', mi);
        verifyEqual(testCase, res.labels, oracle.labels, sprintf('max_iter=%d', mi));
        verifyEqual(testCase, res.cost, oracle.total_cost, 'AbsTol', 1e-12);
    end
end

% =========================================================================
%  Drift 6: the k <= N guard
% =========================================================================

function test_k_above_n_and_no_series_raise_the_cpp_messages(testCase)
    X = testCase.TestData.X;
    err = capture_error(@() dtwc.cluster(X, size(X, 1) + 1));
    verifyEqual(testCase, err.identifier, 'dtwc:invalidArgument');
    verifyEqual(testCase, err.message, ...
        'cluster: k must not exceed the number of series.');
    for none = {dtwc.load(X, 'SkipRows', size(X, 1)), {}}
        err = capture_error(@() dtwc.cluster(none{1}, 2));
        verifyEqual(testCase, err.identifier, 'dtwc:invalidArgument');
        verifyEqual(testCase, err.message, 'cluster: dataset is empty.');
    end
end

% =========================================================================
%  Drift 7: skip_cols / skip_rows on an in-memory source
% =========================================================================

function test_in_memory_skip_cols_is_honoured(testCase)
%   Drift 7: C++ erases the leading columns of an in-memory series; MATLAB
%   silently ignored skip_cols unless the source was a path.
    X = testCase.TestData.X;
    k = testCase.TestData.k;
    skipped = dtwc.cluster(dtwc.load(X, 'SkipCols', 3), k);
    trimmed = dtwc.cluster(X(:, 4:end), k);
    verifyEqual(testCase, skipped.labels, trimmed.labels);
    verifyEqual(testCase, skipped.cost, trimmed.cost, 'AbsTol', 1e-12);
    verifyNotEqual(testCase, skipped.cost, dtwc.cluster(X, k).cost, ...
        'skip_cols must change the distances');
end

function test_in_memory_skip_cols_beyond_the_series_is_rejected(testCase)
%   Drift 7, second half: a SkipCols longer than a series is refused, as C++ and
%   Python refuse it.
    X = testCase.TestData.X;
    err = capture_error(@() dtwc.cluster(dtwc.load(X, 'SkipCols', 99), 2));
    verifyEqual(testCase, err.identifier, 'dtwc:invalidArgument');
    verifyEqual(testCase, err.message, ...
        'load: SkipCols exceeds an in-memory series length.');
end

function test_in_memory_skip_rows_is_honoured(testCase)
    X = testCase.TestData.X;
    dropped = dtwc.cluster(dtwc.load(X, 'SkipRows', 2), 2);
    trimmed = dtwc.cluster(X(3:end, :), 2);
    verifyEqual(testCase, dropped.labels, trimmed.labels);
    verifyEqual(testCase, dropped.cost, trimmed.cost, 'AbsTol', 1e-12);
end

% =========================================================================
%  Drift 3: Result.save writes the dataset's series names
% =========================================================================

function test_result_save_writes_dataset_series_names(testCase)
%   Drift 3: MATLAB hard-coded the ordinal 0..N-1; the C++/CLI writer emits
%   the names DataLoader assigned (1..N for a batch file), so the MATLAB
%   output was not byte-identical to the CLI's.
    f = [tempname '.csv'];
    fid = fopen(f, 'w');
    fprintf(fid, '0,0,0,0\n0,1,0,1\n20,20,20,20\n20,21,20,21\n');
    fclose(fid);
    cleanupFile = onCleanup(@() delete(f));

    outdir = [tempname '_save'];
    cleanupDir = onCleanup(@() remove_directory(outdir));

    res = dtwc.cluster(f, 2);
    res.save(outdir);
    [~, stem, ~] = fileparts(f);
    rows = readcell(fullfile(outdir, [stem '_labels.csv']));
    written = string(cellfun(@num2str, rows(2:end, 1), 'UniformOutput', false));
    verifyEqual(testCase, written, ["1"; "2"; "3"; "4"]);
end

function test_result_distance_matrix_is_the_saved_matrix(testCase)
%   Result.distance_matrix() (and plot()) reads the clustered Problem's matrix:
%   the one the Tier-2 route computes and save() writes (the CSV is
%   max_digits10, so exact), filled on demand after a matrix-free method.
    X = testCase.TestData.X;
    res = dtwc.cluster(X, 2);
    D = res.distance_matrix();

    verifyEqual(testCase, D, dtwc.compute_distance_matrix(X), 'AbsTol', 1e-12);
    verifyEqual(testCase, D, D.');
    onebatch = dtwc.cluster(X, 2, 'Method', 'onebatch');
    verifyEqual(testCase, onebatch.distance_matrix(), D);

    outdir = [tempname '_dm'];
    cleanupDir = onCleanup(@() remove_directory(outdir));
    res.save(outdir);
    saved = readmatrix(fullfile(outdir, 'dataset_distance_matrix.csv'), ...
                       'Delimiter', ',', 'NumHeaderLines', 0);
    verifyEqual(testCase, D, saved);
end

% =========================================================================
%  Files: the same file read the same way in every language (Volkan 10-02)
% =========================================================================

function test_parquet_reads_as_cpp_reads(testCase)
%   The MEX links no Arrow, so MATLAB reads Parquet with its own parquetread and
%   takes what the C++ reader takes: a list column is one series per row, named
%   series_<i>. These are the series pyarrow and the C++ reader read from the
%   tracked fixture (tests/python/test_io.py reads it too).
    here = fileparts(mfilename('fullpath'));
    data = dtwc.load(fullfile(here, '..', 'fixtures', 'fast_clara_streaming_8x4.parquet'));
    [series, names] = data.as_series();
    verifyEqual(testCase, series, {[0 0.1 0 0.2], [0.2 0.1 0.3 0.2], [-0.1 0 0.1 0], ...
        [0.3 0.2 0.4 0.3], [10 10.2 9.9 10.1], [9.8 10 10.1 9.9], [10.3 10.1 10.4 10.2], ...
        [9.7 9.9 9.8 10]});
    verifyEqual(testCase, names, compose('series_%d', 0:7));
    verifyEqual(testCase, data.Name, 'fast_clara_streaming_8x4');
    err = capture_error(@() dtwc.cluster([tempname '.parquet'], 2));   % no such file
    verifyEqual(testCase, err.identifier, 'dtwc:ioError');
end

function test_arrow_ipc_is_refused_naming_the_format(testCase)
%   MATLAB has no Arrow IPC reader (R2026a has no featherread), so an .arrow,
%   .ipc or .feather path is refused before anything is read, never taken as text.
    for ext = {'.arrow', '.ipc', '.feather'}
        err = capture_error(@() dtwc.cluster(['series' ext{1}], 2));
        verifyEqual(testCase, err.identifier, 'dtwc:invalidArgument');
        verifySubstring(testCase, err.message, 'is Arrow IPC, which MATLAB has no reader for');
    end
end

% =========================================================================
%  Drift 2: device is a per-call override, never a global mutation
% =========================================================================

function test_per_call_device_is_validated_and_does_not_leak(testCase)
    dtwc.device('cpu');
    verifyError(testCase, ...
        @() dtwc.cluster(testCase.TestData.X, 2, 'Device', 'not_a_device'), ...
        'dtwc:deviceError');
    verifyEqual(testCase, dtwc.device(), 'cpu');

    res = dtwc.cluster(testCase.TestData.X, 2, 'Device', 'cpu');
    verifyEqual(testCase, res.device, 'cpu');
    verifyEqual(testCase, dtwc.device(), 'cpu');
end

% =========================================================================
%  Drift 9: Problem::checkpoint is bindable from MATLAB
% =========================================================================

function test_checkpoint_options_round_trip(testCase)
    prob = dtwc.Problem('ckpt_roundtrip');
    opts = dtwc.CheckpointOptions('directory', tempname, ...
                                  'save_interval', 7, 'enabled', true);
    prob.set_checkpoint(opts);
    verifyEqual(testCase, prob.get_checkpoint(), opts);

    defaults = dtwc.CheckpointOptions();
    prob.set_checkpoint(defaults);
    verifyEqual(testCase, prob.get_checkpoint(), defaults);
end

function test_checkpoint_mid_fill_publishes_and_resumes(testCase)
%   fill_distance_matrix() consumes Problem::checkpoint (contract 2.7). The
%   checkpoint is the one file <directory>/<name>.dtwm, saved after every
%   save_interval rows and complete after a complete fill.
    X = [0 0 0 0; 0 1 0 1; 20 20 20 20; 20 21 20 21];
    ckdir = [tempname '_ck'];
    cleanupDir = onCleanup(@() remove_directory(ckdir));

    prob = problem_with(X, 'ckpt_midfill');
    prob.set_checkpoint(dtwc.CheckpointOptions( ...
        'directory', ckdir, 'save_interval', 1, 'enabled', true));
    prob.fill_distance_matrix();

    files = dir(ckdir);
    files = {files(~[files.isdir]).name};
    verifyEqual(testCase, files, {'ckpt_midfill.dtwm'}, ...
        'an enabled mid-fill checkpoint is the one <name>.dtwm file, nothing else');

    fresh = problem_with(X, 'ckpt_midfill');
    verifyTrue(testCase, dtwc.load_checkpoint(fresh, ckdir));
    verifyTrue(testCase, fresh.is_distance_matrix_filled(), ...
        'a completed fill must leave a complete checkpoint');
    verifyEqual(testCase, fresh.distance_matrix(), prob.distance_matrix());
end

function test_checkpoint_invalid_settings_surface_cpp_messages(testCase)
    X = testCase.TestData.X;

    zero_interval = problem_with(X, 'ckpt_zero');
    zero_interval.set_checkpoint(dtwc.CheckpointOptions( ...
        'directory', tempname, 'save_interval', 0, 'enabled', true));
    err = capture_error(@() zero_interval.fill_distance_matrix());
    verifyEqual(testCase, err.identifier, 'dtwc:invalidArgument');
    verifyEqual(testCase, err.message, ['Problem::fill_distance_matrix: ' ...
        'checkpoint.save_interval must be at least 1 row when ' ...
        'checkpoint.enabled; got 0.']);

    no_dir = problem_with(X, 'ckpt_nodir');
    no_dir.set_checkpoint(dtwc.CheckpointOptions( ...
        'directory', '', 'save_interval', 1, 'enabled', true));
    err = capture_error(@() no_dir.fill_distance_matrix());
    verifyEqual(testCase, err.identifier, 'dtwc:invalidArgument');
    verifyEqual(testCase, err.message, ['Problem::fill_distance_matrix: ' ...
        'checkpoint.enabled requires a non-empty checkpoint.directory.']);
end

% =========================================================================
%  Drift 10: save_checkpoint / load_checkpoint carry the metric
% =========================================================================

function test_checkpoint_metric_is_part_of_the_identity(testCase)
%   Drift 10: without the metric argument a SquaredL2 matrix was stamped and
%   reloaded as L1 (C++/Python have taken the argument since 2.0).
    X = testCase.TestData.X;
    ckdir = [tempname '_ckm'];
    cleanupDir = onCleanup(@() remove_directory(ckdir));

    prob = problem_with(X, 'ckpt_metric');
    prob.fill_distance_matrix();
    dtwc.save_checkpoint(prob, ckdir, 'squared_euclidean');

    fresh = problem_with(X, 'ckpt_metric');
    verifyError(testCase, @() dtwc.load_checkpoint(fresh, ckdir, 'l1'), ...
        'dtwc:invalidArgument', 'an L1 load must reject a SquaredL2 checkpoint');
    verifyTrue(testCase, dtwc.load_checkpoint(fresh, ckdir, 'squared_euclidean'));

    verifyError(testCase, ...
        @() dtwc.load_checkpoint(fresh, ckdir, 'not_a_metric'), ...
        'dtwc:invalidArgument');
end

function test_checkpoint_absent_file_is_false_not_an_error(testCase)
%   C++ load_checkpoint returns false, with the Problem unchanged, only when
%   there is no such file; every other failure is an exception.
    prob = problem_with(testCase.TestData.X, 'ckpt_absent');
    verifyFalse(testCase, dtwc.load_checkpoint(prob, [tempname '_absent']));
    verifyFalse(testCase, prob.is_distance_matrix_filled());
end

% -------------------------------------------------------------------------
%  Local helpers
% -------------------------------------------------------------------------

function ok = mip_backend_available()
%MIP_BACKEND_AVAILABLE True when an exact MIP solver is compiled into the MEX.
%   Same probe as tests/matlab/test_cluster_mip.m: the exact backend throws
%   when neither HiGHS nor Gurobi is compiled in.
    ok = true;
    try
        p = dtwc.Problem('mip_probe');
        p.set_data([0 0; 1 1]);
        p.set_n_clusters(1);
        p.set_method('mip');
        p.cluster();
    catch
        ok = false;
    end
end

function err = capture_error(fn)
    err = [];
    try
        fn();
    catch caught
        err = caught;
    end
    assert(~isempty(err), 'expected an error, none was raised');
end

function cleaner = scratch_working_directory()
%SCRATCH_WORKING_DIRECTORY cd into a fresh temp directory for the caller's life.
%   The Kmedoids route persists Lloyd artifacts under './results', a path
%   relative to the process working directory; concurrent gate runs collide
%   there (LESSONS F45). Restored, and removed, by the returned onCleanup.
    previous = pwd;
    scratch = [tempname '_dtwc_cwd'];
    mkdir(scratch);
    mkdir(fullfile(scratch, 'results'));  % Lloyd writes there and does not create it
    cd(scratch);
    cleaner = onCleanup(@() restore_working_directory(previous, scratch));
end

function restore_working_directory(previous, scratch)
    cd(previous);
    remove_directory(scratch);
end

function remove_directory(d)
    if exist(d, 'dir')
        rmdir(d, 's');
    end
end
