function tests = test_cluster_mip
%TEST_CLUSTER_MIP Regression test for dtwc.cluster(..., 'method','mip').
%
%   Targets audit finding A12 (.claude/reports/2026-09-02-review-backends-bindings.md):
%   the 'mip' branch of +dtwc/cluster.m called prob.set_method('mip') and
%   prob.cluster() WITHOUT ever calling prob.set_n_clusters(k). Problem::Nc
%   defaults to 1, so the exact MIP backend solved a 1-medoid problem and
%   dtwc.cluster returned ONE cluster for every requested k -- a silent
%   wrong answer on a documented public entry point. Every sibling branch
%   ('pam', 'clara', 'hclust') forwards k.
%
%   HOW THIS FAILS ON THE UNFIXED CODE: numel(unique(res.labels)) == 1 and
%   numel(res.medoids) == 1 for k = 3, so the two verifyNumElements below
%   fail. With set_n_clusters(k) in place the MIP backend returns exactly k
%   medoids and the labels reproduce the three well-separated groups.
%
%   Run with: results = runtests('test_cluster_mip');
%   (Requires the compiled MEX on the path AND a HiGHS-enabled build; the
%    exact MIP backend throws when no MIP solver is compiled in.)
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

% -------------------------------------------------------------------------
%  A12: the MIP branch must honour k
% -------------------------------------------------------------------------

function test_mip_branch_returns_k_clusters(testCase)
%   Unfixed cluster.m leaves Problem::Nc at its default 1 -> one medoid.
    res = dtwc.cluster(testCase.TestData.X, testCase.TestData.k, 'method', 'mip');
    verifyNumElements(testCase, res.medoids, testCase.TestData.k);
    verifyNumElements(testCase, unique(res.labels), testCase.TestData.k);
end

function test_mip_branch_separates_the_three_groups(testCase)
%   With k honoured, the three well-separated triples must not mix.
    res = dtwc.cluster(testCase.TestData.X, testCase.TestData.k, 'method', 'mip');
    lab = double(res.labels(:));
    verifyEqual(testCase, lab(1), lab(2));
    verifyEqual(testCase, lab(2), lab(3));
    verifyEqual(testCase, lab(4), lab(5));
    verifyEqual(testCase, lab(5), lab(6));
    verifyEqual(testCase, lab(7), lab(8));
    verifyEqual(testCase, lab(8), lab(9));
    verifyNotEqual(testCase, lab(1), lab(4));
    verifyNotEqual(testCase, lab(4), lab(7));
end

function test_mip_matches_pam_cluster_count(testCase)
%   Cross-route parity: 'pam' already honours k; 'mip' must agree.
    mip = dtwc.cluster(testCase.TestData.X, testCase.TestData.k, 'method', 'mip');
    pam = dtwc.cluster(testCase.TestData.X, testCase.TestData.k, 'method', 'pam');
    verifyEqual(testCase, numel(unique(mip.labels)), numel(unique(pam.labels)));
end
