function tests = test_pdlp_binding
%TEST_PDLP_BINDING Cross-language parity gate for the PDLP LP-relaxation bound.
%
%   dtwc.pdlp_lp_bound / dtwc.pdlp_gpu_available mirror the C++
%   dtwc::mip::pdlp_lp_bound / pdlp_gpu_available and the Python binding of the
%   same names (report section E2: the PDLP arbiter was C++-only).
%
%   The solve itself needs a HiGHS-enabled MEX; on a HiGHS-less build the
%   binding must raise the typed 'dtwc:solverError' rather than return a
%   fabricated bound, and that loud-failure path IS asserted here.
%
%   Run with: results = runtests('test_pdlp_binding');
    tests = functiontests(localfunctions);
end

function setupOnce(testCase)
    testCase.TestData.mex_available = (exist('dtwc_mex', 'file') == 3);
    % Two clear pairs; the 2-medoid optimum is 2 and the LP relaxation is tight.
    testCase.TestData.D = [0 1 8 9; 1 0 7 8; 8 7 0 1; 9 8 1 0];
end

function setup(testCase)
    assumeTrue(testCase, testCase.TestData.mex_available, ...
        'dtwc_mex MEX unavailable - skipping.');
end

function test_pdlp_gpu_available_is_logical(testCase)
    tf = dtwc.pdlp_gpu_available();
    verifyClass(testCase, tf, 'logical');
    verifyTrue(testCase, isscalar(tf));
end

function test_pdlp_lp_bound_rejects_nonsquare_D(testCase)
    verifyError(testCase, @() dtwc.pdlp_lp_bound([1 2 3; 4 5 6], 2), ...
        'MATLAB:pdlp_lp_bound:expectedSquare');
end

function test_pdlp_lp_bound_rejects_unknown_option(testCase)
    verifyError(testCase, ...
        @() dtwc.pdlp_lp_bound(testCase.TestData.D, 2, 'no_such_option', 1), ...
        'dtwc:invalidArgument');
end

function test_pdlp_lp_bound_rejects_odd_option_list(testCase)
    verifyError(testCase, ...
        @() dtwc.pdlp_lp_bound(testCase.TestData.D, 2, 'tol'), ...
        'dtwc:invalidArgument');
end

function test_pdlp_lp_bound_is_loud_or_correct(testCase)
%   Either HiGHS is compiled in and the bound brackets the optimum (2), or the
%   binding raises the typed solver error. Never a silent zero.
    try
        r = dtwc.pdlp_lp_bound(testCase.TestData.D, 2);
    catch err
        verifyEqual(testCase, err.identifier, 'dtwc:solverError');
        verifySubstring(testCase, err.message, 'HiGHS');
        return;
    end
    verifyTrue(testCase, r.solved);
    verifyLessThanOrEqual(testCase, r.lp_bound, 2 + 1e-6);
    verifyGreaterThanOrEqual(testCase, r.lp_bound, 0);
    verifyClass(testCase, r.gpu_used, 'logical');
end
