function tests = test_mex_input_validation
%TEST_MEX_INPUT_VALIDATION Regression tests for dtwc_mex entry-point guards.
%
%   Targets audit CRITICAL #6 (handoff-2026-06-01-adversarial-audit.md):
%   dtwc_mex.cpp had NO mxIsDouble / mxIsComplex / dimension guard on any
%   entry point. In the R2018a+ interleaved-complex API, mxGetDoubles()
%   returns NULL for a non-double or complex array; the old code fed that
%   NULL straight into the DTW kernels -> NULL-deref -> MATLAB HARD CRASH.
%
%   These tests call the RAW dtwc_mex(...) gateway directly (NOT the +dtwc
%   wrappers, which cast with double() and would mask the bug). They feed
%   int32 / single / logical / complex / empty / N-D / sparse / cell / char
%   inputs and require a clean 'dtwc:invalidArgument' error instead of a crash.
%
%   HOW THIS FAILS ON THE UNFIXED CODE: the very first int32/complex case
%   would NULL-deref inside dtwc::dtwBanded and SEGFAULT the MATLAB process,
%   aborting the whole runtests session (verifyError can never catch a
%   segfault). With the guards in place every case throws cleanly and the
%   suite completes. Positive-control cases confirm the guards do NOT reject
%   valid real-double input.
%
%   Run with: results = runtests('test_mex_input_validation');
%   (Requires the compiled MEX on the path; otherwise every test is SKIPPED
%    with a loud notice -- MATLAB cannot be run in the CI/build environment.)
    tests = functiontests(localfunctions);
end

% -------------------------------------------------------------------------
%  Fixtures: skippable-with-loud-notice when the MEX is not available
% -------------------------------------------------------------------------

function setupOnce(testCase)
    testCase.TestData.mex_available = (exist('dtwc_mex', 'file') == 3); % 3 == MEX-file
    if ~testCase.TestData.mex_available
        bar = repmat('=', 1, 74);
        warning('dtwc:mexNotBuilt', ['\n' bar '\n' ...
            'SKIPPING dtwc_mex input-validation regression tests.\n' ...
            'Reason: compiled gateway ''dtwc_mex'' not found on the MATLAB path.\n' ...
            'These tests exercise C++ entry-point guards and need the built binary.\n' ...
            'To run them:\n' ...
            '    cmake -B build -DDTWC_BUILD_MATLAB=ON && cmake --build build\n' ...
            '    addpath(fullfile(''build'',''bin''));  addpath(''bindings/matlab'');\n' ...
            '    results = runtests(''test_mex_input_validation'')\n' ...
            bar]);
    end
end

function setup(testCase)
    % assumeTrue records an assumption failure (SKIP), not a test failure,
    % when the MEX is unavailable -- this is the "skippable" contract.
    assumeTrue(testCase, testCase.TestData.mex_available, ...
        'dtwc_mex MEX unavailable - skipping (see loud warning above).');
end

% -------------------------------------------------------------------------
%  Stateless distance entry points: to_std_vector / direct mxGetDoubles path
% -------------------------------------------------------------------------

function test_dtw_int32_rejected(testCase)
%   int32 -> mxGetDoubles returns NULL on unfixed code -> NULL-deref crash.
    verifyError(testCase, ...
        @() dtwc_mex('dtw_distance', int32([1 2 3]), int32([1 2 3])), ...
        'dtwc:invalidArgument');
end

function test_dtw_complex_rejected(testCase)
%   complex double survives even the +dtwc double() cast; mxGetDoubles->NULL.
    verifyError(testCase, ...
        @() dtwc_mex('dtw_distance', complex([1 2 3], [1 1 1]), [1 2 3]), ...
        'dtwc:invalidArgument');
end

function test_dtw_single_rejected(testCase)
    verifyError(testCase, ...
        @() dtwc_mex('dtw_distance', single([1 2 3]), single([4 5 6])), ...
        'dtwc:invalidArgument');
end

function test_dtw_logical_rejected(testCase)
    verifyError(testCase, ...
        @() dtwc_mex('dtw_distance', logical([1 0 1]), [1 2 3]), ...
        'dtwc:invalidArgument');
end

function test_dtw_empty_rejected(testCase)
    verifyError(testCase, ...
        @() dtwc_mex('dtw_distance', [], [1 2 3]), ...
        'dtwc:invalidArgument');
end

function test_dtw_struct_rejected(testCase)
%   Non-numeric struct: mxGetDoubles(struct)->NULL on unfixed code.
    verifyError(testCase, ...
        @() dtwc_mex('dtw_distance', struct('a', 1), [1 2 3]), ...
        'dtwc:invalidArgument');
end

function test_dtw_nonnumeric_scalar_rejected(testCase)
%   Valid x,y but a char passed where a numeric band is expected (get_scalar).
    verifyError(testCase, ...
        @() dtwc_mex('dtw_distance', [1 2 3], [1 2 3], 'ten'), ...
        'dtwc:invalidArgument');
end

function test_soft_dtw_int32_rejected(testCase)
%   Covers the to_std_vector() guard used by soft_dtw / missing / arow.
    verifyError(testCase, ...
        @() dtwc_mex('soft_dtw_distance', int32([1 2 3]), int32([1 2 3])), ...
        'dtwc:invalidArgument');
end

% -------------------------------------------------------------------------
%  Matrix entry points: matrix_to_series path (set_data / distance matrix)
% -------------------------------------------------------------------------

function test_distance_matrix_int32_rejected(testCase)
    verifyError(testCase, ...
        @() dtwc_mex('compute_distance_matrix', int32([1 2 3; 4 5 6])), ...
        'dtwc:invalidArgument');
end

function test_distance_matrix_complex_rejected(testCase)
    verifyError(testCase, ...
        @() dtwc_mex('compute_distance_matrix', complex(ones(2,3), ones(2,3))), ...
        'dtwc:invalidArgument');
end

function test_distance_matrix_ndarray_rejected(testCase)
%   3-D double array: unfixed matrix_to_series would silently misread it;
%   the dimension guard now rejects any N-D (ndim != 2) input.
    verifyError(testCase, ...
        @() dtwc_mex('compute_distance_matrix', ones(2, 3, 2)), ...
        'dtwc:invalidArgument');
end

function test_distance_matrix_sparse_rejected(testCase)
%   Sparse double: mxGetDoubles returns the compressed-column nonzeros with a
%   layout matrix_to_series misinterprets -> silent-wrong / OOB on unfixed code.
    verifyError(testCase, ...
        @() dtwc_mex('compute_distance_matrix', sparse(eye(3))), ...
        'dtwc:invalidArgument');
end

% -------------------------------------------------------------------------
%  Handle-based entry point: Problem_set_distance_matrix
% -------------------------------------------------------------------------

function test_set_distance_matrix_complex_rejected(testCase)
    h = dtwc_mex('Problem_new', 'validation_test');
    guard = onCleanup(@() dtwc_mex('Problem_delete', h)); %#ok<NASGU>
    dtwc_mex('Problem_set_data', h, [1 2 3; 4 5 6]);   % 2 series of length 3
    badDM = complex(ones(2, 2), ones(2, 2));
    verifyError(testCase, ...
        @() dtwc_mex('Problem_set_distance_matrix', h, badDM), ...
        'dtwc:invalidArgument');
end

% -------------------------------------------------------------------------
%  Label entry points: ARI / NMI accept int32 OR double, reject anything else
% -------------------------------------------------------------------------

function test_ari_complex_labels_rejected(testCase)
    verifyError(testCase, ...
        @() dtwc_mex('adjusted_rand_index', complex([1 2], [0 1]), [1 2]), ...
        'dtwc:invalidArgument');
end

function test_ari_cell_labels_rejected(testCase)
%   Cell is neither int32 nor double: unfixed else-branch mxGetDoubles->NULL.
    verifyError(testCase, ...
        @() dtwc_mex('adjusted_rand_index', {1, 2}, [1 2]), ...
        'dtwc:invalidArgument');
end

function test_nmi_single_labels_rejected(testCase)
    verifyError(testCase, ...
        @() dtwc_mex('normalized_mutual_information', single([1 2 1]), [1 2 1]), ...
        'dtwc:invalidArgument');
end

% -------------------------------------------------------------------------
%  Positive controls: valid real double must STILL work (guards not too strict)
% -------------------------------------------------------------------------

function test_valid_double_vector_still_works(testCase)
    d = dtwc_mex('dtw_distance', [1 2 3 4], [1 2 3 4]);
    verifyEqual(testCase, d, 0, 'AbsTol', 1e-12);   % self-distance == 0
end

function test_valid_double_matrix_still_works(testCase)
    D = dtwc_mex('compute_distance_matrix', [1 2 3; 4 5 6]);
    verifySize(testCase, D, [2 2]);
    verifyEqual(testCase, diag(D), zeros(2, 1), 'AbsTol', 1e-12);
end

function test_valid_int32_labels_still_work(testCase)
%   ARI/NMI must keep accepting int32 labels (documented dual acceptance).
    v = dtwc_mex('adjusted_rand_index', int32([1 2 1 2]), int32([1 2 1 2]));
    verifyEqual(testCase, v, 1, 'AbsTol', 1e-12);   % identical labelings -> ARI 1
end

end
