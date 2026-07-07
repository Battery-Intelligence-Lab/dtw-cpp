function tests = test_conformance
%TEST_CONFORMANCE MATLAB route of the cross-language conformance fixture (Task 2.4).
%
%   The permanent parity gate for docs/api-contract-2.0.md §9. Drives the LIVE
%   +dtwc / dtwc_mex pipeline (Problem.set_band -> fill_distance_matrix ->
%   fast_pam -> silhouette/davies_bouldin/dunn) on the recorded conformance
%   dataset and asserts digit-identical labels/medoids and scores within 1e-12
%   rel against conformance_reference.txt (recorded by the C++ route).
%
%   Fixed pipeline (in lockstep with cpp_conformance.cpp, conformance.toml and
%   test_conformance.py): load conformance_series.csv -> banded DTW (band=3) ->
%   FastPAM k=3 -> scores. Labels/medoids are canonicalised (sorted medoid SET;
%   each point labelled by the rank of its assigned medoid) and converted from
%   MATLAB's 1-based MEX boundary to 0-based series indices before comparison.
%
%   Run with:  results = runtests('test_conformance');
%   (Requires the compiled dtwc_mex on the path; otherwise the test is SKIPPED
%    with a loud notice, matching test_contract_parity.m.)
    tests = functiontests(localfunctions);
end

% -------------------------------------------------------------------------
function setupOnce(testCase)
    testCase.TestData.mex_available = (exist('dtwc_mex', 'file') == 3); % 3 == MEX-file
    if ~testCase.TestData.mex_available
        bar = repmat('=', 1, 74);
        warning('dtwc:mexNotBuilt', ['\n' bar '\n' ...
            'SKIPPING dtwc cross-language conformance test.\n' ...
            'Reason: compiled gateway ''dtwc_mex'' not found on the MATLAB path.\n' ...
            'Build with: cmake -B build -DDTWC_BUILD_MATLAB=ON && cmake --build build\n' ...
            'then addpath(build bin) and addpath(''bindings/matlab'').\n' bar]);
    end
    here = fileparts(mfilename('fullpath'));
    testCase.TestData.data_csv  = fullfile(here, 'data', 'conformance_series.csv');
    testCase.TestData.reference = fullfile(here, 'conformance_reference.txt');
    if testCase.TestData.mex_available
        dtwc.device('cpu');
    end
end

function setup(testCase)
    assumeTrue(testCase, testCase.TestData.mex_available, ...
        'dtwc_mex MEX unavailable - skipping (see loud warning above).');
end

% =========================================================================
function test_matlab_route_matches_reference(testCase)
%   Drives dtwc.fast_pam Tier-2 + scores on the recorded conformance dataset.
    N_CLUSTERS = 3;
    BAND = 3;
    MAX_ITER = 100;
    ref = read_reference(testCase.TestData.reference);

    X = readmatrix(testCase.TestData.data_csv);   % 27 x 16, rows = series

    prob = dtwc.Problem('conformance');
    prob.set_data(X);
    prob.set_band(BAND);
    prob.fill_distance_matrix();
    res = dtwc.fast_pam(prob, N_CLUSTERS, 'MaxIter', MAX_ITER); % writes back into prob

    % --- canonicalise (MEX is 1-based; reference is 0-based) ---
    raw_labels  = double(res.labels(:))';           % 1-based index into medoids array
    raw_medoids = double(res.medoid_indices(:))';   % 1-based series indices
    [labels, medoids] = canonicalise(raw_labels, raw_medoids);

    % --- scores (permutation invariant; read prob state written by fast_pam) ---
    sil = dtwc.silhouette(prob);
    silhouette = mean(sil(:));
    davies_bouldin = dtwc.davies_bouldin(prob);
    dunn = dtwc.dunn(prob);

    % Labels + medoids: DIGIT-IDENTICAL.
    verifyEqual(testCase, labels, ref.labels, 'labels differ from reference');
    verifyEqual(testCase, medoids, ref.medoids, 'medoids differ from reference');
    verifyEqual(testCase, numel(medoids), N_CLUSTERS);

    % Scores: equal to 1e-12 relative.
    verify_rel(testCase, silhouette, ref.silhouette, 'silhouette');
    verify_rel(testCase, davies_bouldin, ref.davies_bouldin, 'davies_bouldin');
    verify_rel(testCase, dunn, ref.dunn, 'dunn');
end

% -------------------------------------------------------------------------
function [labels, medoids] = canonicalise(raw_labels, raw_medoids)
%CANONICALISE Sorted 0-based medoid SET + each point labelled by the rank of its
%   assigned medoid. Mirrors cpp_conformance.cpp / test_conformance.py exactly.
    med0 = raw_medoids - 1;                    % 1-based -> 0-based series indices
    medoids = sort(med0);                      % canonical sorted medoid set
    assigned0 = med0(raw_labels);              % 0-based series index of each point's medoid
    labels = zeros(1, numel(assigned0));
    for i = 1:numel(assigned0)
        labels(i) = find(medoids == assigned0(i), 1) - 1;   % 0-based rank
    end
end

function ref = read_reference(path)
    ref = struct();
    lines = readlines(path);
    for i = 1:numel(lines)
        line = strtrim(lines(i));
        if line == "" || startsWith(line, "#"); continue; end
        parts = split(line, ",");
        key = char(parts(1));
        if strcmp(key, 'labels') || strcmp(key, 'medoids')
            ref.(key) = str2double(parts(2:end))';   % 0-based int row vector
        else
            ref.(key) = str2double(parts(2));
        end
    end
end

function verify_rel(testCase, actual, expected, name)
    tol = 1e-12 * max(abs(actual), abs(expected));
    verifyLessThanOrEqual(testCase, abs(actual - expected), tol, ...
        sprintf('%s: %.17g vs reference %.17g exceeds 1e-12 rel', name, actual, expected));
end
