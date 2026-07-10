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

function test_variant_parameter_domains_are_typed(testCase)
%   M34: malformed recurrence parameters must fail at the MEX boundary rather
%   than entering arithmetic (or taking an identity/empty shortcut).
    badCalls = {
        @() dtwc_mex('wdtw_distance', [0], [0 0], -1, -1), ...
        @() dtwc_mex('adtw_distance', [0], [0 0], -1, -1), ...
        @() dtwc_mex('soft_dtw_distance', [0], [0 0], 0), ...
        @() dtwc_mex('soft_dtw_gradient', [0], [0 0], NaN)
    };
    for i = 1:numel(badCalls)
        verifyError(testCase, badCalls{i}, 'dtwc:invalidArgument');
    end
end

function test_variant_wrappers_preserve_dtwc_error_type(testCase)
%   Wrapper-side validation uses the same public identifier as the raw MEX
%   gateway, so callers do not see inputParser-specific error types.
    verifyError(testCase, ...
        @() dtwc.distance.wdtw([0], [0 0], 'G', -1), ...
        'dtwc:invalidArgument');
    verifyError(testCase, ...
        @() dtwc.distance.adtw([0], [0 0], 'Penalty', -1), ...
        'dtwc:invalidArgument');
    verifyError(testCase, ...
        @() dtwc.distance.soft_dtw([0], [0 0], 'Gamma', 0), ...
        'dtwc:invalidArgument');
end

function test_variant_zero_and_near_zero_boundaries_are_valid(testCase)
    verifyEqual(testCase, dtwc.distance.wdtw([0], [0 0], 'G', 0), 0, ...
        'AbsTol', 0);
    verifyEqual(testCase, dtwc.distance.adtw([0], [0 0], 'Penalty', 0), 0, ...
        'AbsTol', 0);
    d = dtwc.distance.soft_dtw([0], [0 0], 'Gamma', realmin('double'));
    verifyTrue(testCase, isfinite(d));
end

function test_unknown_metric_tokens_are_invalid_arguments(testCase)
%   M36: wrappers must distinguish an unknown token from a known-but-
%   unsupported metric and must never run the L1 kernel as a substitute.
    calls = {
        @() dtwc.distance.standard([0 3], [0 1], 'Metric', 'bogus'), ...
        @() dtwc.distance.missing([0 3], [0 1], 'Metric', 'bogus'), ...
        @() dtwc.distance.arow([0 3], [0 1], 'Metric', 'bogus')
    };
    for i = 1:numel(calls)
        verifyError(testCase, calls{i}, 'dtwc:invalidArgument');
    end
end

function test_problem_rejects_variant_missing_cross_product(testCase)
    h = dtwc_mex('Problem_new', 'm36_cross_product');
    guard = onCleanup(@() dtwc_mex('Problem_delete', h)); %#ok<NASGU>
    dtwc_mex('Problem_set_data', h, [0 0; 0 0]);
    dtwc_mex('Problem_set_variant', h, 'adtw', 0.75);
    verifyError(testCase, ...
        @() dtwc_mex('Problem_set_missing_strategy', h, 'zero_cost'), ...
        'dtwc:invalidArgument');
end

function test_matlab_dispatch_rejects_variant_missing_cross_product(testCase)
    verifyError(testCase, ...
        @() dtwc.distance.dtw([0], [0 0], ...
            'Variant', 'adtw', 'MissingStrategy', 'zero_cost'), ...
        'dtwc:invalidArgument');
end

function test_rejected_missing_setter_preserves_complete_cache(testCase)
%   M48: the whole-method setter must validate before clearing published work.
    prob = m48_problem();
    prob.set_variant('adtw', 2.0);
    prob.set_distance_matrix([0 123; 123 0]);
    verifyTrue(testCase, prob.is_distance_matrix_filled());

    verifyError(testCase, @() prob.set_missing_strategy('zero_cost'), ...
        'dtwc:invalidArgument');
    verifyTrue(testCase, prob.is_distance_matrix_filled());
end

function test_rejected_missing_setter_preserves_exact_cache_values(testCase)
    prob = m48_problem();
    expected = [0 123; 123 0];
    prob.set_variant('adtw', 2.0);
    prob.set_distance_matrix(expected);

    verifyError(testCase, @() prob.set_missing_strategy('zero_cost'), ...
        'dtwc:invalidArgument');
    % No selector getter exists in MATLAB. Restoring through the public method
    % is a no-op only when the rejected candidate was never published.
    prob.set_missing_strategy('error');
    verifyEqual(testCase, prob.distance_matrix(), expected, 'AbsTol', 0);
end

function test_rejected_variant_setter_preserves_complete_cache(testCase)
    prob = m48_problem();
    prob.set_missing_strategy('zero_cost');
    prob.set_distance_matrix([0 456; 456 0]);
    verifyTrue(testCase, prob.is_distance_matrix_filled());

    verifyError(testCase, @() prob.set_variant('adtw', 7.5), ...
        'dtwc:invalidArgument');
    verifyTrue(testCase, prob.is_distance_matrix_filled());
end


function test_rejected_variant_setter_preserves_exact_cache_values(testCase)
    prob = m48_problem();
    expected = [0 456; 456 0];
    prob.set_missing_strategy('zero_cost');
    prob.set_distance_matrix(expected);

    verifyError(testCase, @() prob.set_variant('adtw', 7.5), ...
        'dtwc:invalidArgument');
    prob.set_variant('standard');
    verifyEqual(testCase, prob.distance_matrix(), expected, 'AbsTol', 0);
end

function test_semantic_setter_noops_preserve_matlab_cache(testCase)
    adtw = m48_problem();
    adtw.set_variant('adtw', 2.0);
    adtw.set_distance_matrix([0 321; 321 0]);
    adtw.set_missing_strategy('error');
    adtw.set_variant('adtw', 2.0);
    verifyTrue(testCase, adtw.is_distance_matrix_filled());
    verifyEqual(testCase, adtw.distance_matrix(), [0 321; 321 0], 'AbsTol', 0);

    missing = m48_problem();
    missing.set_missing_strategy('zero_cost');
    missing.set_distance_matrix([0 654; 654 0]);
    missing.set_variant('standard');
    verifyTrue(testCase, missing.is_distance_matrix_filled());
    verifyEqual(testCase, missing.distance_matrix(), [0 654; 654 0], 'AbsTol', 0);
end

function test_valid_semantic_setters_publish_and_invalidate_matlab_cache(testCase)
    missing = m48_problem();
    missing.set_distance_matrix([0 111; 111 0]);
    missing.set_missing_strategy('zero_cost');
    verifyFalse(testCase, missing.is_distance_matrix_filled());

    variant = m48_problem();
    variant.set_distance_matrix([0 222; 222 0]);
    variant.set_variant('adtw', 2.0);
    verifyFalse(testCase, variant.is_distance_matrix_filled());
end

function test_invalid_cuda_precision_values_are_typed(testCase)
%   M47: validate the exact integer selector before static_cast<int>, Problem
%   publication, cache invalidation, or backend capability selection.
    invalid = {-1, 3, NaN, Inf, 1.5, double(intmax('int32')) + 1};
    for i = 1:numel(invalid)
        h = dtwc_mex('Problem_new', 'm47_cuda_precision');
        guard = onCleanup(@() dtwc_mex('Problem_delete', h)); %#ok<NASGU>
        dtwc_mex('Problem_set_data', h, [0; 1]);
        verifyError(testCase, ...
            @() dtwc_mex('Problem_set_cuda_settings', h, 0, invalid{i}), ...
            'dtwc:invalidArgument');
        clear guard;
    end
end

function test_unknown_problem_selector_tokens_remain_typed(testCase)
%   Text parsers are an independent first boundary; M47 must not weaken their
%   established invalidArgument behavior while hardening raw C++ enum values.
    h = dtwc_mex('Problem_new', 'm47_unknown_selectors');
    guard = onCleanup(@() dtwc_mex('Problem_delete', h)); %#ok<NASGU>
    calls = {
        @() dtwc_mex('Problem_set_variant', h, 'bogus'), ...
        @() dtwc_mex('Problem_set_missing_strategy', h, 'bogus'), ...
        @() dtwc_mex('Problem_set_distance_strategy', h, 'bogus'), ...
        @() dtwc_mex('Problem_set_lb_strategy', h, 'bogus'), ...
        @() dtwc_mex('Problem_set_storage_policy', h, 'bogus')
    };
    for i = 1:numel(calls)
        verifyError(testCase, calls{i}, 'dtwc:invalidArgument');
    end
end

function prob = m48_problem()
    prob = dtwc.Problem('m48_matlab');
    prob.set_data([0; 1]);
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
