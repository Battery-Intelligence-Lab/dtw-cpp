function tests = test_mex_input_validation
%TEST_MEX_INPUT_VALIDATION Regression tests for dtwc_mex entry-point guards.
%
%   Why these guards exist:
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

function test_invalid_cuda_device_id_conversion_is_typed(testCase)
%   M47: reject values with no defined C++ int conversion. Negative exact
%   integers remain backend-policy inputs and are intentionally not covered.
    invalid = {NaN, Inf, 1.5, double(intmax('int32')) + 1};
    for i = 1:numel(invalid)
        h = dtwc_mex('Problem_new', 'm47_cuda_device_id');
        guard = onCleanup(@() dtwc_mex('Problem_delete', h)); %#ok<NASGU>
        verifyError(testCase, ...
            @() dtwc_mex('Problem_set_cuda_settings', h, invalid{i}, 0), ...
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
        @() dtwc_mex('Problem_set_distance_strategy', h, 'bogus')
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
        @() dtwc_mex('adjusted_rand', complex([1 2], [0 1]), [1 2]), ...
        'dtwc:invalidArgument');
end

function test_ari_cell_labels_rejected(testCase)
%   Cell is neither int32 nor double: unfixed else-branch mxGetDoubles->NULL.
    verifyError(testCase, ...
        @() dtwc_mex('adjusted_rand', {1, 2}, [1 2]), ...
        'dtwc:invalidArgument');
end

function test_nmi_single_labels_rejected(testCase)
    verifyError(testCase, ...
        @() dtwc_mex('normalized_mutual_info', single([1 2 1]), [1 2 1]), ...
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
    v = dtwc_mex('adjusted_rand', int32([1 2 1 2]), int32([1 2 1 2]));
    verifyEqual(testCase, v, 1, 'AbsTol', 1e-12);   % identical labelings -> ARI 1
end

% -------------------------------------------------------------------------
%  A13: mx_to_dendrogram validated only the ROW count of 'merges'
%  A14: label vectors were cast float->int without a finiteness check
% -------------------------------------------------------------------------

function test_dendrogram_three_column_merges_rejected(testCase)
%   The reader indexes column 4 (data[i + 3*n_merges]). With only 3 columns the
%   unfixed code read past the end of the mxArray's heap buffer and fed the
%   garbage into Dendrogram::new_size.
    h = dtwc_mex('Problem_new', 'dend_cols_test');
    guard = onCleanup(@() dtwc_mex('Problem_delete', h)); %#ok<NASGU>
    dtwc_mex('Problem_set_data', h, [1 2 3; 4 5 6; 7 8 9]);
    dend = struct('merges', [1 2 0.5; 3 4 1.5], 'n_points', int32(3));
    verifyError(testCase, ...
        @() dtwc_mex('cut_dendrogram', dend, h, 2), ...
        'dtwc:invalidArgument');
end

function test_dendrogram_transposed_merges_rejected(testCase)
%   A 4xM transposed 'merges' has the right element count but the wrong shape.
    h = dtwc_mex('Problem_new', 'dend_transposed_test');
    guard = onCleanup(@() dtwc_mex('Problem_delete', h)); %#ok<NASGU>
    dtwc_mex('Problem_set_data', h, [1 2 3; 4 5 6; 7 8 9]);
    dend = struct('merges', [1 2 0.5 2; 3 4 1.5 3]', 'n_points', int32(3));
    verifyError(testCase, ...
        @() dtwc_mex('cut_dendrogram', dend, h, 2), ...
        'dtwc:invalidArgument');
end

function test_dendrogram_nonfinite_merge_rejected(testCase)
%   NaN in an index column: static_cast<int>(NaN) is undefined behaviour.
    h = dtwc_mex('Problem_new', 'dend_nan_test');
    guard = onCleanup(@() dtwc_mex('Problem_delete', h)); %#ok<NASGU>
    dtwc_mex('Problem_set_data', h, [1 2 3; 4 5 6; 7 8 9]);
    dend = struct('merges', [1 2 0.5 2; NaN 4 1.5 3], 'n_points', int32(3));
    verifyError(testCase, ...
        @() dtwc_mex('cut_dendrogram', dend, h, 2), ...
        'dtwc:invalidArgument');
end

function test_ari_nan_labels_rejected(testCase)
%   require_label_vector accepted NaN; static_cast<int>(NaN - 1) is UB.
    verifyError(testCase, ...
        @() dtwc_mex('adjusted_rand', [1 NaN 2 1], [1 2 2 1]), ...
        'dtwc:invalidArgument');
end

function test_nmi_inf_labels_rejected(testCase)
    verifyError(testCase, ...
        @() dtwc_mex('normalized_mutual_info', [1 2 1 2], [1 Inf 1 2]), ...
        'dtwc:invalidArgument');
end

function test_ari_fractional_labels_rejected(testCase)
%   A fractional label is not a cluster id; silently truncating it changed the
%   score instead of reporting the caller's mistake.
    verifyError(testCase, ...
        @() dtwc_mex('adjusted_rand', [1 1.5 2 1], [1 2 2 1]), ...
        'dtwc:invalidArgument');
end

function test_valid_double_dendrogram_still_cuts(testCase)
%   Positive control: a well-formed Nx4 'merges' must still be accepted.
    h = dtwc_mex('Problem_new', 'dend_positive_test');
    guard = onCleanup(@() dtwc_mex('Problem_delete', h)); %#ok<NASGU>
    dtwc_mex('Problem_set_data', h, [0 0; 0 1; 20 20]);
    dtwc_mex('Problem_fill_distance_matrix', h);
    dend = dtwc_mex('build_dendrogram', h, 'average', 100);
    res = dtwc_mex('cut_dendrogram', dend, h, 2);
    verifyNumElements(testCase, res.medoid_indices, 2);
    verifyNumElements(testCase, res.labels, 3);
end

function test_labels_are_values_not_indices(testCase)
%   A cluster label is a name, not a position: 0, negative and INT_MIN labels
%   are labels (C++ and Python take them), and a relabelling never changes the
%   score. Only a non-integer (NaN, Inf, fractional) is refused, above.
    verifyEqual(testCase, dtwc_mex('adjusted_rand', [0 0 1 1], [1 1 2 2]), 1, 'AbsTol', 1e-12);
    verifyEqual(testCase, dtwc_mex('normalized_mutual_info', [0 0 1 1], [1 1 2 2]), 1, 'AbsTol', 1e-12);
    verifyEqual(testCase, dtwc_mex('adjusted_rand', [-3 -3 7 7], [5 5 -1 -1]), 1, 'AbsTol', 1e-12);
    verifyEqual(testCase, dtwc.adjusted_rand([0 0 1 1], [1 1 2 2]), 1, 'AbsTol', 1e-12);
    int_min = -2147483648;
    expected = dtwc_mex('adjusted_rand', [1 3 2 1], [1 2 2 1]);
    verifyEqual(testCase, dtwc_mex('adjusted_rand', [1 int_min 2 1], [1 2 2 1]), expected, 'AbsTol', 1e-12);
    verifyEqual(testCase, dtwc_mex('adjusted_rand', int32([1 int_min 2 1]), int32([1 2 2 1])), ...
        expected, 'AbsTol', 1e-12);
end

% -------------------------------------------------------------------------
%  Gateway errors and handle lifetime
% -------------------------------------------------------------------------

function test_invalid_handle_and_unknown_command_are_typed(testCase)
    verifyError(testCase, @() dtwc_mex('Problem_get_size', uint64(99999)), ...
        'dtwc:invalidArgument');
    verifyError(testCase, @() dtwc_mex('nonexistent_command'), ...
        'dtwc:invalidArgument');
end

function test_deleted_problem_handle_is_invalid(testCase)
    prob = dtwc.Problem('delete_test');
    h = prob.get_handle();
    verifyGreaterThan(testCase, h, 0);
    delete(prob);
    verifyError(testCase, @() dtwc_mex('Problem_get_size', uint64(h)), ...
        'dtwc:invalidArgument');
end

% -------------------------------------------------------------------------
%  Integer arguments are read exactly. A fractional, NaN or Inf value, and an
%  index below 1, is dtwc:invalidArgument; each was truncated or cast with
%  undefined behaviour.
% -------------------------------------------------------------------------

function test_integer_arguments_reject_fractions_nan_and_inf(testCase)
    h = int_problem(testCase);
    h2 = dtwc_mex('Problem_new', 'exact_int_ndim');
    testCase.addTeardown(@() dtwc_mex('Problem_delete', h2));
    x = [1 2 3 4 5];
    y = [2 3 4 5 6];
    X = [1 2 3 4 5; 2 3 4 5 6; 9 8 7 6 5; 8 7 6 5 4];
    dend = dtwc_mex('build_dendrogram', h, 'average', 100);
    sites = {
        'Problem_set_band',       @(v) dtwc_mex('Problem_set_band', h, v),                3
        'Problem_set_max_iter',   @(v) dtwc_mex('Problem_set_max_iter', h, v),            5
        'Problem_set_n_repetition', @(v) dtwc_mex('Problem_set_n_repetition', h, v),      1
        'Problem_set_n_clusters', @(v) dtwc_mex('Problem_set_n_clusters', h, v),          2
        'Problem_set_data ndim',  @(v) dtwc_mex('Problem_set_data', h2, X(1:2, 1:4), {}, v), 2
        'time_limit_sec',         @(v) dtwc_mex('Problem_set_mip_settings', h, struct('time_limit_sec', v)), 30
        'numeric_focus',          @(v) dtwc_mex('Problem_set_mip_settings', h, struct('numeric_focus', v)), 0
        'mip_focus',              @(v) dtwc_mex('Problem_set_mip_settings', h, struct('mip_focus', v)), 0
        'dtw_distance band',      @(v) dtwc_mex('dtw_distance', x, y, v),                 3
        'ddtw_distance band',     @(v) dtwc_mex('ddtw_distance', x, y, v),                3
        'wdtw_distance band',     @(v) dtwc_mex('wdtw_distance', x, y, v),                3
        'adtw_distance band',     @(v) dtwc_mex('adtw_distance', x, y, v),                3
        'dtw_distance_missing band', @(v) dtwc_mex('dtw_distance_missing', x, y, v),      3
        'dtw_arow_distance band', @(v) dtwc_mex('dtw_arow_distance', x, y, v),            3
        'compute_distance_matrix band', @(v) dtwc_mex('compute_distance_matrix', X, v),   3
        'fast_pam k',             @(v) dtwc_mex('fast_pam', h, v),                        2
        'fast_pam max_iter',      @(v) dtwc_mex('fast_pam', h, 2, v),                     5
        'fast_clara k',           @(v) dtwc_mex('fast_clara', h, v),                      2
        'fast_clara sample_size', @(v) dtwc_mex('fast_clara', h, 2, v),                   3
        'fast_clara n_samples',   @(v) dtwc_mex('fast_clara', h, 2, 3, v),                2
        'fast_clara max_iter',    @(v) dtwc_mex('fast_clara', h, 2, 3, 2, v),             5
        'build_dendrogram max_points', @(v) dtwc_mex('build_dendrogram', h, 'average', v), 100
        'cut_dendrogram k',       @(v) dtwc_mex('cut_dendrogram', dend, h, v),            2
        'cluster k',              @(v) dtwc_mex('cluster', X, v),                         2
        'cluster band',           @(v) dtwc_mex('cluster', X, 2, v),                      3
        'cluster max_iter',       @(v) dtwc_mex('cluster', X, 2, -1, 0, v),               5
    };
    for i = 1:size(sites, 1)
        call = sites{i, 2};
        call(sites{i, 3});   % an exact integer passes
        for bad = {2.5, NaN, Inf, -Inf}
            verifyError(testCase, @() call(bad{1}), 'dtwc:invalidArgument', ...
                sprintf('%s accepted %g', sites{i, 1}, bad{1}));
        end
    end
end

function test_index_arguments_must_be_at_least_one(testCase)
    h = int_problem(testCase);
    for bad = {0, -1, 2.5, NaN, Inf}
        verifyError(testCase, @() dtwc_mex('Problem_dist_by_ind', h, bad{1}, 1), ...
            'dtwc:invalidArgument', sprintf('i = %g', bad{1}));
        verifyError(testCase, @() dtwc_mex('Problem_dist_by_ind', h, 1, bad{1}), ...
            'dtwc:invalidArgument', sprintf('j = %g', bad{1}));
    end
    verifyGreaterThan(testCase, dtwc_mex('Problem_dist_by_ind', h, 1, 3), 0);
    % A dendrogram merge id is an index too.
    dend = dtwc_mex('build_dendrogram', h, 'average', 100);
    verifyNumElements(testCase, dtwc_mex('cut_dendrogram', dend, h, 2).labels, 4);
    dend.merges(1, 1) = 0;
    verifyError(testCase, @() dtwc_mex('cut_dendrogram', dend, h, 2), ...
        'dtwc:invalidArgument');
end

function test_dist_by_ind_index_above_n_is_an_error(testCase)
%   Problem::dist_by_ind is the unchecked hot path and returned 0 for an index
%   past the last series; the boundary owns the check and names index and N.
    h = int_problem(testCase);   % N = 4
    verifyGreaterThan(testCase, dtwc_mex('Problem_dist_by_ind', h, 4, 1), 0);   % N is valid
    for bad = {5, 100}
        verifyError(testCase, @() dtwc_mex('Problem_dist_by_ind', h, bad{1}, 1), ...
            'dtwc:invalidArgument', sprintf('i = %g', bad{1}));
        verifyError(testCase, @() dtwc_mex('Problem_dist_by_ind', h, 1, bad{1}), ...
            'dtwc:invalidArgument', sprintf('j = %g', bad{1}));
    end
    err = [];
    try
        dtwc_mex('Problem_dist_by_ind', h, 1, 5);
    catch err
    end
    verifyNotEmpty(testCase, err, 'j = 5 was accepted');
    verifySubstring(testCase, err.message, 'j = 5');
    verifySubstring(testCase, err.message, 'N = 4');
end

function test_a_double_handle_must_be_an_exact_integer(testCase)
    h = int_problem(testCase);
    verifyEqual(testCase, dtwc_mex('Problem_get_size', double(h)), 4);
    for bad = {double(h) + 0.5, NaN, Inf, -1}
        verifyError(testCase, @() dtwc_mex('Problem_get_size', bad{1}), ...
            'dtwc:invalidArgument', sprintf('handle %g', bad{1}));
    end
end

function h = int_problem(testCase)
%   A filled four-series Problem the integer-argument sites can be called on.
    h = dtwc_mex('Problem_new', 'exact_int');
    testCase.addTeardown(@() dtwc_mex('Problem_delete', h));
    dtwc_mex('Problem_set_data', h, ...
        [1 2 3 4 5; 2 3 4 5 6; 9 8 7 6 5; 8 7 6 5 4]);
    dtwc_mex('Problem_fill_distance_matrix', h);
end

% -------------------------------------------------------------------------
%  Names are parsed by dtwc::parse_name over the C++ tables (the CLI's),
%  ignoring ASCII case, so MATLAB accepts every spelling C++ does.
% -------------------------------------------------------------------------

function test_every_spelling_in_the_cpp_name_tables_is_accepted(testCase)
    h = dtwc_mex('Problem_new', 'names');
    testCase.addTeardown(@() dtwc_mex('Problem_delete', h));
    for name = {'kmedoids', 'mip', 'lrcore', 'tadpole', 'LRCore'}
        dtwc_mex('Problem_set_method', h, name{1});
    end
    for name = {'standard', 'ddtw', 'wdtw', 'adtw', 'softdtw', 'soft-dtw', 'msm', ...
                'twe', 'MSM'}
        dtwc_mex('Problem_set_variant', h, name{1});
    end
    dtwc_mex('Problem_set_variant', h, 'standard');
    for name = {'error', 'zero_cost', 'zero-cost', 'zerocost', 'arow', ...
                'interpolate', 'AROW'}
        dtwc_mex('Problem_set_missing_strategy', h, name{1});
    end
    for name = {'highs', 'gurobi', 'HiGHS'}
        dtwc_mex('Problem_set_solver', h, name{1});   % false: not compiled in
    end
    dtwc_mex('Problem_set_data', h, [1 2 3; 2 3 4; 9 8 7; 8 7 6]);
    for name = {'single', 'complete', 'average', 'Complete'}
        dend = dtwc_mex('build_dendrogram', h, name{1}, 100);
        verifySize(testCase, dend.merges, [3 4]);
    end
end

function test_metric_names_share_one_meaning(testCase)
%   'l2sq' and 'SqEuclidean' were unknown to MATLAB; the table spells them.
    X = [1 2 3; 2 3 5; 9 8 7; 8 7 5];
    squared = dtwc_mex('DTWClustering_compute_distance_matrix', X, -1, 'squared_euclidean');
    for name = {'sqeuclidean', 'l2sq', 'SqEuclidean'}
        verifyEqual(testCase, ...
            dtwc_mex('DTWClustering_compute_distance_matrix', X, -1, name{1}), squared);
    end
    verifyNotEqual(testCase, ...
        dtwc_mex('DTWClustering_compute_distance_matrix', X, -1, 'l1'), squared);
end

function test_unknown_names_list_the_cpp_table(testCase)
%   The "Valid:" list is parse_name's: canonical spellings, in table order.
    h = dtwc_mex('Problem_new', 'names_unknown');
    testCase.addTeardown(@() dtwc_mex('Problem_delete', h));
    dtwc_mex('Problem_set_data', h, [1 2 3; 2 3 4; 9 8 7; 8 7 6]);
    cases = {
        @() dtwc_mex('Problem_set_method', h, 'bogus'), ...
            'unknown method ''bogus''. Valid: kmedoids, mip, lrcore, tadpole.'
        @() dtwc_mex('Problem_set_solver', h, 'bogus'), ...
            'unknown solver ''bogus''. Valid: highs, gurobi.'
        @() dtwc_mex('build_dendrogram', h, 'bogus', 100), ...
            'unknown linkage ''bogus''. Valid: single, complete, average.'
        @() dtwc_mex('Problem_set_missing_strategy', h, 'bogus'), ...
            ['unknown missing strategy ''bogus''. Valid: error, zero_cost, arow, ' ...
             'interpolate.']
        @() dtwc_mex('DTWClustering_compute_distance_matrix', [1 2; 3 4], -1, 'bogus'), ...
            'unknown metric ''bogus''. Valid: l1, squared_euclidean.'
    };
    for i = 1:size(cases, 1)
        err = [];
        try
            cases{i, 1}();
        catch caught
            err = caught;
        end
        assertNotEmpty(testCase, err);
        verifyEqual(testCase, err.identifier, 'dtwc:invalidArgument');
        verifyEqual(testCase, err.message, cases{i, 2});
    end
end
