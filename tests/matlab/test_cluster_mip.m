function tests = test_cluster_mip
%TEST_CLUSTER_MIP Contract test for dtwc.cluster(..., 'Method', 'mip').
%
%   Requires the compiled MEX on the path and a HiGHS-enabled build; the
%   exact MIP backend throws when no MIP solver is compiled in.

%> @file test_cluster_mip.m
%> @author Volkan Kumtepeli
    tests = functiontests(localfunctions);
end

% -------------------------------------------------------------------------
%  Fixtures
% -------------------------------------------------------------------------

function setupOnce(testCase)
    testCase.TestData.mex_available = (exist('dtwc_mex', 'file') == 3); % 3 == MEX-file

    % Three well-separated groups of three length-4 series. The optimal
    % 3-medoid clustering is unambiguous, so the assertions below do not
    % depend on tie-breaking.
    testCase.TestData.X = [ ...
        0 0 0 0; 0 1 0 1; 1 0 1 0; ...
        20 20 20 20; 20 21 20 21; 21 20 21 20; ...
        40 40 40 40; 40 41 40 41; 41 40 41 40];
    testCase.TestData.k = 3;

    % Capability probe: the exact MIP route needs HiGHS (or Gurobi) compiled
    % into the MEX. Probe once, loudly, so an unavailable solver is never
    % confused with a wrong answer.
    testCase.TestData.mip_available = false;
    if testCase.TestData.mex_available
        try
            p = dtwc.Problem('mip_probe');
            p.set_data([0 0; 1 1]);
            p.set_n_clusters(1);
            p.set_method('mip');
            p.cluster();
            testCase.TestData.mip_available = true;
        catch probeErr
            bar = repmat('=', 1, 74);
            warning('dtwc:mipUnavailable', ['\n' bar '\n' ...
                'SKIPPING dtwc.cluster MIP tests: no exact MIP solver in this MEX.\n' ...
                'Probe error: %s\n' ...
                'Rebuild with -DDTWC_ENABLE_HIGHS=ON -DDTWC_BUILD_MATLAB=ON.\n' ...
                bar], probeErr.message);
        end
    end
end

function setup(testCase)
    assumeTrue(testCase, testCase.TestData.mex_available, ...
        'dtwc_mex MEX unavailable - skipping.');
    assumeTrue(testCase, testCase.TestData.mip_available, ...
        'No exact MIP solver compiled into dtwc_mex - skipping.');
end

function test_mip_branch_separates_the_three_groups(testCase)
    res = dtwc.cluster(testCase.TestData.X, testCase.TestData.k, 'Method', 'mip');
    lab = double(res.labels(:));
    verifyNumElements(testCase, res.medoids, testCase.TestData.k);
    verifyNumElements(testCase, unique(lab), testCase.TestData.k);
    verifyEqual(testCase, lab(1), lab(2));
    verifyEqual(testCase, lab(2), lab(3));
    verifyEqual(testCase, lab(4), lab(5));
    verifyEqual(testCase, lab(5), lab(6));
    verifyEqual(testCase, lab(7), lab(8));
    verifyEqual(testCase, lab(8), lab(9));
    verifyNotEqual(testCase, lab(1), lab(4));
    verifyNotEqual(testCase, lab(4), lab(7));
end
