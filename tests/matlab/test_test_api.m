function tests = test_test_api
%TEST_TEST_API dtwc.test introspection API parity gate (Task 3.3).
%
%   Drives the LIVE public entry points dtwc_mex('test_parallelisation') /
%   dtwc_mex('test_gpu') AND the thin wrappers dtwc.test.parallelisation() /
%   dtwc.test.gpu() (bindings/matlab/+dtwc/+test/*.m), asserting the SAME field
%   names as the C++ (tests/unit/test_test_api.cpp) and Python
%   (tests/python/test_test_api.py) suites.
%
%   REGISTERED EXPECTATION (release MEX uses a validated OpenMP runtime):
%     test_parallelisation -> available=true, pass=true, reason empty and
%                             threads_engaged>=2 on the multicore release gate;
%     test_gpu             -> available=false, validated=false, pass=false,
%                             reason non-empty naming the missing GPU backend.
%
%   Run with: results = runtests('test_test_api');
%   (Requires the compiled dtwc_mex on the path; otherwise every test is SKIPPED
%    with a loud notice.)
    tests = functiontests(localfunctions);
end

function test_version_matches_ssot(testCase)
    repoRoot = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    expected = strtrim(fileread(fullfile(repoRoot, 'VERSION')));
    verifyEqual(testCase, dtwc_mex('version'), expected);
end

% -------------------------------------------------------------------------
%  Fixtures
% -------------------------------------------------------------------------

function setupOnce(testCase)
    testCase.TestData.mex_available = (exist('dtwc_mex', 'file') == 3); % 3 == MEX-file
    if ~testCase.TestData.mex_available
        bar = repmat('=', 1, 74);
        warning('dtwc:mexNotBuilt', ['\n' bar '\n' ...
            'SKIPPING dtwc.test introspection tests.\n' ...
            'Reason: compiled gateway ''dtwc_mex'' not found on the MATLAB path.\n' ...
            'Build with: cmake -B build -DDTWC_BUILD_MATLAB=ON -DDTWC_ALLOW_SEQUENTIAL=ON\n' ...
            'then addpath(build bin) and addpath(''bindings/matlab'').\n' bar]);
    end
end

function setup(testCase)
    assumeTrue(testCase, testCase.TestData.mex_available, ...
        'dtwc_mex MEX unavailable - skipping (see loud warning above).');
end

% =========================================================================
%  parallelisation() — schema + honest serial answer
% =========================================================================

function test_parallelisation_schema(testCase)
%TEST_PARALLELISATION_SCHEMA dtwc_mex('test_parallelisation') field names.
    r = dtwc_mex('test_parallelisation');
    verifyTrue(testCase, isstruct(r));
    expected = {'available', 'max_threads', 'threads_engaged', 'pass', 'reason'};
    verifyEqual(testCase, sort(fieldnames(r))', sort(expected));
    verifyTrue(testCase, islogical(r.available));
    verifyTrue(testCase, islogical(r.pass));
    verifyGreaterThanOrEqual(testCase, r.max_threads, 1);
    verifyGreaterThanOrEqual(testCase, r.threads_engaged, 1);
end

function test_parallelisation_runtime_engages(testCase)
%TEST_PARALLELISATION_RUNTIME_ENGAGES Release MEX runs a real parallel region.
    r = dtwc_mex('test_parallelisation');
    verifyTrue(testCase, r.available);
    verifyTrue(testCase, r.pass);
    verifyEmpty(testCase, r.reason);
    verifyGreaterThanOrEqual(testCase, r.threads_engaged, 2);
end

function test_parallelisation_wrapper_matches_mex(testCase)
%TEST_PARALLELISATION_WRAPPER_MATCHES_MEX Pins dtwc.test.parallelisation().
    r = dtwc.test.parallelisation();
    verifyTrue(testCase, isstruct(r));
    verifyTrue(testCase, r.available);
    verifyTrue(testCase, r.pass);
    verifyEmpty(testCase, r.reason);
    verifyGreaterThanOrEqual(testCase, r.threads_engaged, 2);
end

% =========================================================================
%  gpu() — schema + honest unavailable answer (never throws, never silent)
% =========================================================================

function test_gpu_schema(testCase)
%TEST_GPU_SCHEMA dtwc_mex('test_gpu') field names.
    r = dtwc_mex('test_gpu');
    verifyTrue(testCase, isstruct(r));
    expected = {'available', 'backend', 'device_name', 'validated', 'pass', 'reason'};
    verifyEqual(testCase, sort(fieldnames(r))', sort(expected));
    verifyTrue(testCase, islogical(r.available));
    verifyTrue(testCase, islogical(r.validated));
    verifyTrue(testCase, islogical(r.pass));
end

function test_gpu_unavailable_is_honest(testCase)
%TEST_GPU_UNAVAILABLE_IS_HONEST Registered: CUDA-OFF build -> available=false.
%   Pins dtwc_mex('test_gpu'). No throw, no silent degrade — a non-empty reason
%   naming the missing GPU backend.
    r = dtwc_mex('test_gpu');
    verifyFalse(testCase, r.available);
    verifyFalse(testCase, r.validated);
    verifyFalse(testCase, r.pass);
    verifyNotEmpty(testCase, r.reason);
end

function test_gpu_wrapper_matches_mex(testCase)
%TEST_GPU_WRAPPER_MATCHES_MEX Pins dtwc.test.gpu().
    r = dtwc.test.gpu();
    verifyTrue(testCase, isstruct(r));
    verifyFalse(testCase, r.available);
    verifyNotEmpty(testCase, r.reason);
end
