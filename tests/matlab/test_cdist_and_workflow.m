function tests = test_cdist_and_workflow
%TEST_CDIST_AND_WORKFLOW MATLAB tests for dtwc.cdist and settings/load_data/do_clustering/write_data.
%
%   Exercises rectangular cross-collection DTW distances (column-major MATLAB
%   matrices, cell arrays, Band, Metric, Variant, MatrixDtype='float32') and
%   the CasADi-style workflow across persistent output/checkpoint and pure
%   in-memory modes.

%> @file test_cdist_and_workflow.m
%> @author Volkan Kumtepeli
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

    D = dtwc.cdist(X, Y, 'Band', 3, 'Metric', 'squared_euclidean');
    verifySize(testCase, D, [5 11]);
    verifyClass(testCase, D, 'double');
    for i = 1:size(X, 1)
        for j = 1:size(Y, 1)
            ref = dtwc.distance.dtw(X(i, :), Y(j, :), 'Band', 3, 'Metric', 'squared_euclidean');
            verifyEqual(testCase, D(i, j), ref, 'RelTol', 1e-12);
        end
    end

    % Self-cdist (Y = []) matches dtwc.cdist(X, X, ...)
    D_self = dtwc.distance.cdist(X, [], 'Band', 3, 'Metric', 'squared_euclidean');
    D_xx = dtwc.cdist(X, X, 'Band', 3, 'Metric', 'squared_euclidean');
    verifyEqual(testCase, D_self, D_xx, 'RelTol', 1e-12);

    % Empty non-omitted Y collections return [n_X, 0] rather than triggering self-cdist
    verifySize(testCase, dtwc.cdist(X, zeros(0, 12)), [5 0]);
    verifySize(testCase, dtwc.cdist(X, {}), [5 0]);

    % Float32 output matrix
    D_f32 = dtwc.cdist(X, Y, 'Band', 3, 'Metric', 'squared_euclidean', 'MatrixDtype', 'float32');
    verifyClass(testCase, D_f32, 'single');
    verifyEqual(testCase, D_f32, single(D));
end

function test_cdist_ragged_cells_and_multivariate(testCase)
    X_cell = {[0 1 2 1], [3 3.5 4 4.2 3.8 3.1]};
    Y_cell = {[0.2 0.9 1.8 1.2 0.5], [3.1 3.6 3.9 4.1], [-1 -2 -1.5]};
    D = dtwc.cdist(X_cell, Y_cell, 'Variant', 'wdtw', 'WdtwG', 0.07);
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
    D_mv = dtwc.cdist(X_mv, Y_mv, 'Ndim', 2, 'MvMode', 'dependent', 'Band', 2);
    verifySize(testCase, D_mv, [4 3]);
end

function test_casadi_workflow_persistent_and_in_memory(testCase)
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

    cfg = dtwc.settings('NClusters', 2, 'Method', 'pam', 'Band', 2, ...
                        'SkipRows', 1, 'SkipCols', 1, ...
                        'Output', auto_out, 'Checkpoint', ckpt_dir, 'Name', 'sensor');
    ds = dtwc.load_data(csv_file, cfg);
    res = dtwc.do_clustering(ds, cfg);

    verifyNumElements(testCase, res.labels, 6);
    verifyEqual(testCase, res.labels(1), res.labels(2));
    verifyEqual(testCase, res.labels(1), res.labels(3));
    verifyNotEqual(testCase, res.labels(1), res.labels(4));
    verifyTrue(testCase, isfile(fullfile(auto_out, 'sensor_labels.csv')));
    verifyTrue(testCase, isfile(fullfile(auto_out, 'sensor_medoids.csv')));
    verifyTrue(testCase, isfile(fullfile(auto_out, 'sensor_report.txt')));

    % Pure in-memory override (Output = '') + explicit write_data
    mem_res = dtwc.do_clustering(ds, cfg, 'Output', '', 'Checkpoint', '');
    verifyEqual(testCase, mem_res.labels, res.labels);
    verifyEqual(testCase, mem_res.cost, res.cost, 'RelTol', 1e-12);

    % A Dataset is read with the options it was made with: settings that give one another value are
    % refused, naming it; the same value (above) or the default runs.
    try
        dtwc.do_clustering(ds, cfg, 'Output', '', 'Checkpoint', '', 'SkipRows', 2);
        verifyFail(testCase, 'SkipRows 2 beside a Dataset made with SkipRows 1 ran');
    catch err
        verifyEqual(testCase, err.identifier, 'dtwc:invalidArgument');
        verifySubstring(testCase, err.message, 'SkipRows 1 and the settings give SkipRows 2');
    end
    def_res = dtwc.do_clustering(ds, cfg, 'Output', '', 'Checkpoint', '', 'SkipRows', 0);
    verifyEqual(testCase, def_res.labels, res.labels);
    verifyError(testCase, @() dtwc.cdist(ds, 'Dtype', 'float32'), 'dtwc:invalidArgument');
    verifyError(testCase, @() dtwc.cdist(dtwc.load_data(csv_file, cfg, 'Dtype', 'float32'), 'Dtype', 'float64'), ...
                'dtwc:invalidArgument');

    dtwc.write_data(mem_res, manual_out);
    verifyTrue(testCase, isfile(fullfile(manual_out, 'sensor_labels.csv')));
    verifyTrue(testCase, isfile(fullfile(manual_out, 'sensor_medoids.csv')));

    % Post-init struct field mutation on settings is reconciled and validated
    mut = dtwc.settings('NClusters', 3, 'Method', 'kmedoids', 'Band', -1, 'Output', '', 'Checkpoint', '');
    mut.NClusters = 2;
    mut.Method = 'pam';
    mut.Band = 1;
    mut_res = dtwc.do_clustering(ds, mut);
    verifyNumElements(testCase, mut_res.medoids, 2);

    bad = mut;
    bad.UnknownField = 42;
    verifyError(testCase, @() dtwc.do_clustering(ds, bad), 'dtwc:invalidArgument');
end

