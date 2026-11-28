function tests = test_cdist_and_workflow
%TEST_CDIST_AND_WORKFLOW MATLAB tests for dtwc.distance.cdist and for the files dtwc.cluster writes.
%
%   Exercises rectangular cross-collection DTW distances (column-major MATLAB
%   matrices, cell arrays, Band, Metric, Variant, MatrixDtype='float32') and
%   dtwc.cluster's Output and Checkpoint: the result files and the checkpoint it
%   is asked for, and none otherwise.

%> @file test_cdist_and_workflow.m
%> @brief MATLAB tests for dtwc.distance.cdist and for the files dtwc.cluster writes.
%> @author Volkan Kumtepeli
%> @date 13 Nov 2026
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    testCase.TestData.mex_available = (exist('dtwc_mex', 'file') == 3);
    if testCase.TestData.mex_available
        dtwc.device('cpu');
    end
end

function setup(testCase)
    assumeTrue(testCase, testCase.TestData.mex_available, 'dtwc_mex unavailable');
end

function test_cdist_rectangular_and_self_match_oracle(testCase)
    rng(17);
    X = randn(5, 12);
    Y = 2 + randn(11, 12);

    D = dtwc.distance.cdist(X, Y, 'Band', 3, 'Metric', 'squared_euclidean');
    verifySize(testCase, D, [5 11]);
    verifyClass(testCase, D, 'double');
    for i = 1:size(X, 1)
        for j = 1:size(Y, 1)
            ref = dtwc.distance.dtw(X(i, :), Y(j, :), 'Band', 3, 'Metric', 'squared_euclidean');
            verifyEqual(testCase, D(i, j), ref, 'RelTol', 1e-12);
        end
    end

    % Self-cdist (Y = []) matches dtwc.distance.cdist(X, X, ...)
    D_self = dtwc.distance.cdist(X, [], 'Band', 3, 'Metric', 'squared_euclidean');
    D_xx = dtwc.distance.cdist(X, X, 'Band', 3, 'Metric', 'squared_euclidean');
    verifyEqual(testCase, D_self, D_xx, 'RelTol', 1e-12);

    % Empty non-omitted Y collections return [n_X, 0] rather than triggering self-cdist
    verifySize(testCase, dtwc.distance.cdist(X, zeros(0, 12)), [5 0]);
    verifySize(testCase, dtwc.distance.cdist(X, {}), [5 0]);

    % Float32 output matrix
    D_f32 = dtwc.distance.cdist(X, Y, 'Band', 3, 'Metric', 'squared_euclidean', 'MatrixDtype', 'float32');
    verifyClass(testCase, D_f32, 'single');
    verifyEqual(testCase, D_f32, single(D));
end

function test_cdist_ragged_cells_and_multivariate(testCase)
    X_cell = {[0 1 2 1], [3 3.5 4 4.2 3.8 3.1]};
    Y_cell = {[0.2 0.9 1.8 1.2 0.5], [3.1 3.6 3.9 4.1], [-1 -2 -1.5]};
    D = dtwc.distance.cdist(X_cell, Y_cell, 'Variant', 'wdtw', 'WdtwG', 0.07);
    verifySize(testCase, D, [2 3]);
    for i = 1:numel(X_cell)
        for j = 1:numel(Y_cell)
            ref = dtwc.distance.dtw(X_cell{i}, Y_cell{j}, 'Variant', 'wdtw', 'WdtwG', 0.07);
            verifyEqual(testCase, D(i, j), ref, 'RelTol', 1e-12);
        end
    end

    % Multivariate interleaved (12 cols = 6 timesteps x 2 channels)
    rng(23);
    X_mv = randn(4, 12);
    Y_mv = randn(3, 12);
    D_mv = dtwc.distance.cdist(X_mv, Y_mv, 'Ndim', 2, 'MvMode', 'dependent', 'Band', 2);
    verifySize(testCase, D_mv, [4 3]);

    % dtwc.cluster clusters a Dataset in its Ndim: what Tier 2 gives the series at 2 values a step, not what 1
    % value a step gives
    mv = dtwc.load(X_mv, 'Ndim', 2);
    prob = dtwc.Problem('mv_oracle');
    prob.set_data(X_mv, {}, 2);
    oracle = dtwc.fast_pam(prob, 2);
    verifyEqual(testCase, dtwc.cluster(mv, 2, 'Method', 'pam').cost, oracle.total_cost, 'RelTol', 1e-12);
    verifyNotEqual(testCase, dtwc.cluster(X_mv, 2, 'Method', 'pam').cost, oracle.total_cost);

    % cdist omits Ndim by default (adopts Dataset Ndim), and refuses explicit 'Ndim', 1 on multivariate Dataset
    verifyEqual(testCase, dtwc.distance.cdist(mv), dtwc.distance.cdist(X_mv, 'Ndim', 2), 'RelTol', 1e-12);
    verifyError(testCase, @() dtwc.distance.cdist(mv, 'Ndim', 1), 'dtwc:invalidArgument');
end

function test_compute_distance_matrix_metric_and_device(testCase)
    rng(31);
    X = randn(5, 10);
    D_sq = dtwc.compute_distance_matrix(X, 'Metric', 'squared_euclidean', 'Band', 2, 'Device', 'cpu');
    D_ref = dtwc.distance.cdist(X, X, 'Metric', 'squared_euclidean', 'Band', 2);
    verifyEqual(testCase, D_sq, D_ref, 'RelTol', 1e-12);
    verifyError(testCase, @() dtwc.compute_distance_matrix(X, 'Device', 'bogus_dev'), 'dtwc:deviceError');
end

function test_cluster_writes_the_files_and_checkpoint_it_is_asked_for(testCase)
    work = fullfile(tempdir, ['dtwc_wf_' num2str(feature('getpid')) '_' num2str(randi(1e6))]);
    mkdir(work);
    cleanup = onCleanup(@() rmdir(work, 's')); %#ok<NASGU>

    csv_file = fullfile(work, 'sensor.csv');
    fid = fopen(csv_file, 'w');
    fprintf(fid, 'id,t0,t1,t2,t3\n');
    fprintf(fid, 'a,0.0,0.1,0.2,0.1\n');
    fprintf(fid, 'b,0.1,0.0,0.1,0.2\n');
    fprintf(fid, 'c,0.05,0.15,0.1,0.05\n');
    fprintf(fid, 'd,9.0,9.1,9.2,9.0\n');
    fprintf(fid, 'e,9.2,9.0,9.1,9.3\n');
    fprintf(fid, 'f,8.9,9.1,9.0,9.2\n');
    fclose(fid);

    auto_out = fullfile(work, 'auto_out');
    ckpt_dir = fullfile(work, 'ckpt');
    manual_out = fullfile(work, 'manual_out');

    ds = dtwc.load(csv_file, 'SkipRows', 1, 'SkipCols', 1, 'Name', 'sensor');
    opts = {'Method', 'pam', 'Band', 2};   % kept for reuse: a pair after them overrides one
    res = dtwc.cluster(ds, 2, opts{:}, 'Output', auto_out, 'Checkpoint', ckpt_dir);

    verifyNumElements(testCase, res.labels, 6);
    verifyEqual(testCase, res.labels(1), res.labels(2));
    verifyEqual(testCase, res.labels(1), res.labels(3));
    verifyNotEqual(testCase, res.labels(1), res.labels(4));
    verifyTrue(testCase, isfile(fullfile(auto_out, 'sensor_labels.csv')));
    verifyTrue(testCase, isfile(fullfile(auto_out, 'sensor_medoids.csv')));
    verifyTrue(testCase, isfile(fullfile(auto_out, 'sensor_report.txt')));
    verifyTrue(testCase, isfile(fullfile(ckpt_dir, 'sensor.dtwm')));

    % The same checkpoint resumes.
    resumed = dtwc.cluster(ds, 2, opts{:}, 'Checkpoint', ckpt_dir);
    verifyNotEmpty(testCase, regexp(resumed.report, 'the checkpoint\s+resumed', 'once'));
    verifyEqual(testCase, resumed.labels, res.labels);

    % No Output (the default) writes nothing; Result.save writes on request.
    before = numel(dir(fullfile(work, '**', '*')));
    mem_res = dtwc.cluster(ds, 2, opts{:});
    verifyEqual(testCase, numel(dir(fullfile(work, '**', '*'))), before);
    verifyEqual(testCase, mem_res.labels, res.labels);
    verifyEqual(testCase, mem_res.cost, res.cost, 'RelTol', 1e-12);
    mem_res.save(manual_out);
    verifyTrue(testCase, isfile(fullfile(manual_out, 'sensor_labels.csv')));
    verifyTrue(testCase, isfile(fullfile(manual_out, 'sensor_medoids.csv')));

    % A CheckpointInterval saves a checkpoint, so it needs one.
    verifyError(testCase, @() dtwc.cluster(ds, 2, opts{:}, 'CheckpointInterval', 5), 'dtwc:invalidArgument');

    % A Dataset is read with the options it was made with: dtwc.load refuses one given that differs, at its
    % default too, naming both values; the same value runs.
    try
        dtwc.load(ds, 'SkipRows', 2);
        verifyFail(testCase, 'SkipRows 2 beside a Dataset made with SkipRows 1 was accepted');
    catch err
        verifyEqual(testCase, err.identifier, 'dtwc:invalidArgument');
        verifySubstring(testCase, err.message, 'SkipRows 1 and SkipRows 2 was given');
    end
    verifyError(testCase, @() dtwc.load(ds, 'SkipRows', 0), 'dtwc:invalidArgument');
    verifySameHandle(testCase, dtwc.load(ds, 'SkipRows', 1), ds);
    verifyError(testCase, @() dtwc.distance.cdist(ds, 'Dtype', 'float32'), 'dtwc:invalidArgument');
    ds32 = dtwc.load(csv_file, 'SkipRows', 1, 'SkipCols', 1, 'Dtype', 'float32', 'Name', 'sensor');
    verifyError(testCase, @() dtwc.distance.cdist(ds32, 'Dtype', 'float64'), 'dtwc:invalidArgument');

    % dtwc.load and dtwc.cluster take a Dataset by the same rule: a Dtype that differs is refused, naming
    % both values, at its default too; the same one, or none, runs.
    try
        dtwc.load(ds32, 'Dtype', 'float64');
        verifyFail(testCase, 'Dtype float64 beside a Dataset made with Dtype float32 ran');
    catch err
        verifyEqual(testCase, err.identifier, 'dtwc:invalidArgument');
        verifySubstring(testCase, err.message, 'Dtype ''float32'' and Dtype ''float64'' was given');
    end
    verifySameHandle(testCase, dtwc.load(ds32, 'Dtype', 'float32'), ds32);
    verifySameHandle(testCase, dtwc.load(ds32), ds32);
    verifyError(testCase, @() dtwc.cluster(ds32, 2, 'Dtype', 'float64'), 'dtwc:invalidArgument');
    verifyNumElements(testCase, dtwc.cluster(ds32, 2, 'Dtype', 'float32').labels, 6);
    verifyNumElements(testCase, dtwc.cluster(ds32, 2).labels, 6);
end

function test_a_name_given_twice_takes_its_last_value(testCase)
%   Options kept in a cell array are overridden by a pair after them: dtwc.cluster(X, k, opts{:}, 'Band', 6).
    rng(5);
    X = cumsum(randn(8, 20), 2);
    narrow = dtwc.cluster(X, 2, 'Method', 'pam', 'Band', 1);
    wide = dtwc.cluster(X, 2, 'Method', 'pam', 'Band', 6);
    verifyNotEqual(testCase, narrow.cost, wide.cost);   % the two bands differ on these series
    opts = {'Method', 'pam', 'Band', 1};
    verifyEqual(testCase, dtwc.cluster(X, 2, opts{:}, 'Band', 6).cost, wide.cost);
end
