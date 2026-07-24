function summary = f19_problem_writeback_oracle(route, profile, version)
%F19_PROBLEM_WRITEBACK_ORACLE Prove MATLAB result writeback is core-owned.
%
%   SUMMARY = F19_PROBLEM_WRITEBACK_ORACLE(ROUTE, PROFILE, VERSION) drives
%   the public MATLAB wrappers against a caller-selected MEX. ROUTE is one
%   of "all", "fast_pam", "fast_clara", "clarans", or "cut_dendrogram".
%
%   Every selected route starts from a fresh Problem whose labels, medoids,
%   and k are first poisoned by a deterministic k=3 FastPAM result. The
%   target k=2 command must replace that state. Each execution performs the
%   registered minimum of eight exact vector assertions:
%     - poison returned labels and medoids match literal int32 oracles;
%     - poison stored labels and medoids equal the returned state;
%     - target returned labels and medoids match literal int32 oracles;
%     - target stored labels and medoids equal the returned state.
%
%   PROFILE and VERSION are provenance labels printed in the route markers.
%   This function does not select or build a MEX; the PowerShell runner owns
%   exact path selection and fresh-artifact provenance.

    if nargin ~= 3
        error('dtwc:f19WritebackOracle', ...
            'Expected route, profile, and version arguments.');
    end

    route = lower(strtrim(char(route)));
    profile = char(profile);
    version = char(version);
    validRoutes = {'fast_pam', 'fast_clara', 'clarans', 'cut_dendrogram'};
    if strcmp(route, 'all')
        selectedRoutes = validRoutes;
    elseif any(strcmp(route, validRoutes))
        selectedRoutes = {route};
    else
        error('dtwc:f19WritebackOracle', ...
            'Unknown route ''%s''.', route);
    end
    if isempty(profile) || isempty(version)
        error('dtwc:f19WritebackOracle', ...
            'Profile and version labels must be nonempty.');
    end
    observedRelease = ['R' char(builtin('version', '-release'))];
    if ~strcmp(observedRelease, version)
        error('dtwc:f19WritebackOracle', ...
            'MATLAB release mismatch: requested=%s observed=%s.', ...
            version, observedRelease);
    end
    version = observedRelease;

    % All 28 off-diagonal L1-DTW distances are distinct. Exhaustive k=2
    % enumeration has the unique optimum medoid set {2,7}, cost 87, with a
    % next-best cost of 89 and no optimal assignment tie.
    X = [ ...
         0  1  0  2; ...
         1  0  2  1; ...
         3  5  2  4; ...
         6  2  7  3; ...
        20 21 19 22; ...
        22 18 25 20; ...
        26 25 29 23; ...
        31 28 27 35];

    minimumVectorAssertions = 8;
    vectorAssertions = 0;
    scalarAssertions = 0;
    executions = 0;

    dtwc.device('cpu');
    for i = 1:numel(selectedRoutes)
        result = run_route(selectedRoutes{i}, X, profile, version);
        vectorAssertions = vectorAssertions + result.vector_assertions;
        scalarAssertions = scalarAssertions + result.scalar_assertions;
        executions = executions + 1;
    end

    expectedVectorAssertions = executions * minimumVectorAssertions;
    expectedScalarAssertions = executions * 5;
    assert_scalar(vectorAssertions, expectedVectorAssertions, ...
        'aggregate vector assertion count');
    assert_scalar(scalarAssertions, expectedScalarAssertions, ...
        'aggregate scalar assertion count');

    summary = struct( ...
        'routes', {selectedRoutes}, ...
        'executions', executions, ...
        'vector_assertions', vectorAssertions, ...
        'minimum_vector_assertions', minimumVectorAssertions, ...
        'scalar_assertions', scalarAssertions);
end

function routeSummary = run_route(route, X, profile, version)
    poisonLabels = int32([2 2 2 2 1 1 1 3]);
    poisonMedoids = int32([6 2 8]);
    poisonCost = 53;

    switch route
        case 'fast_pam'
            expectedLabels = int32([2 2 2 2 1 1 1 1]);
            expectedMedoids = int32([7 2]);
            expectedCost = 87;
        case 'fast_clara'
            expectedLabels = int32([2 2 2 2 1 1 1 1]);
            expectedMedoids = int32([7 4]);
            expectedCost = 98;
        case {'clarans', 'cut_dendrogram'}
            expectedLabels = int32([1 1 1 1 2 2 2 2]);
            expectedMedoids = int32([2 7]);
            expectedCost = 87;
        otherwise
            error('dtwc:f19WritebackOracle', ...
                'Unreachable route ''%s''.', route);
    end

    prob = dtwc.Problem(['f19_' route]);
    cleanup = onCleanup(@() delete_problem(prob));
    prob.set_data(X);

    poison = dtwc.fast_pam(prob, 3, 'MaxIter', 100, 'Seed', 29);
    vectorAssertions = 0;
    scalarAssertions = 0;

    vectorAssertions = vectorAssertions + assert_vector( ...
        poison.labels, poisonLabels, [route ' poison labels']);
    vectorAssertions = vectorAssertions + assert_vector( ...
        poison.medoid_indices, poisonMedoids, [route ' poison medoids']);
    vectorAssertions = vectorAssertions + assert_vector( ...
        prob.labels(), poison.labels, [route ' stored poison labels']);
    vectorAssertions = vectorAssertions + assert_vector( ...
        prob.medoids(), poison.medoid_indices, ...
        [route ' stored poison medoids']);
    scalarAssertions = scalarAssertions + assert_scalar( ...
        poison.total_cost, poisonCost, [route ' poison cost']);
    scalarAssertions = scalarAssertions + assert_scalar( ...
        double(prob.n_clusters()), 3, [route ' poison k']);

    switch route
        case 'fast_pam'
            target = dtwc.fast_pam( ...
                prob, 2, 'MaxIter', 100, 'Seed', 42);
        case 'fast_clara'
            % SampleSize < N is load-bearing: it exercises FastCLARA's own
            % resident writeback body rather than its full-data FastPAM
            % delegate. The resulting medoids [7 4], cost 98 distinguish it.
            target = dtwc.fast_clara( ...
                prob, 2, 'SampleSize', 4, 'NSamples', 3, ...
                'MaxIter', 100, 'Seed', 42);
        case 'clarans'
            target = dtwc.clarans( ...
                prob, 2, 'NumLocal', 3, 'MaxNeighbor', 50, ...
                'MaxDtwEvals', -1, 'Seed', 42);
        case 'cut_dendrogram'
            prob.fill_distance_matrix();
            dend = dtwc.build_dendrogram( ...
                prob, 'Linkage', 'average', 'MaxPoints', 8);
            target = dtwc.cut_dendrogram(dend, prob, 2);
    end

    vectorAssertions = vectorAssertions + assert_vector( ...
        target.labels, expectedLabels, [route ' result labels']);
    vectorAssertions = vectorAssertions + assert_vector( ...
        target.medoid_indices, expectedMedoids, [route ' result medoids']);
    vectorAssertions = vectorAssertions + assert_vector( ...
        prob.labels(), target.labels, [route ' stored labels']);
    vectorAssertions = vectorAssertions + assert_vector( ...
        prob.medoids(), target.medoid_indices, [route ' stored medoids']);

    scalarAssertions = scalarAssertions + assert_scalar( ...
        target.total_cost, expectedCost, [route ' result cost']);
    scalarAssertions = scalarAssertions + assert_scalar( ...
        double(prob.n_clusters()), 2, [route ' result k']);
    scalarAssertions = scalarAssertions + assert_scalar( ...
        prob.find_total_cost(), expectedCost, [route ' stored-state cost']);

    minimumVectorAssertions = 8;
    assert_scalar(vectorAssertions, minimumVectorAssertions, ...
        [route ' vector assertion count']);
    assert_scalar(scalarAssertions, 5, [route ' scalar assertion count']);

    fprintf(['F19_MATLAB_ROUTE profile=%s version=%s ' ...
        'observed_release=%s route=%s ' ...
        'executions=1 vector_assertions=%d/%d ' ...
        'minimum_vector_assertions=%d scalar_assertions=%d/5 ' ...
        'labels=%s medoids=%s cost=%.17g stored=1/1 k=2 skips=0\n'], ...
        profile, version, version, route, vectorAssertions, ...
        minimumVectorAssertions, minimumVectorAssertions, ...
        scalarAssertions, mat2str(target.labels), ...
        mat2str(target.medoid_indices), target.total_cost);

    routeSummary = struct( ...
        'vector_assertions', vectorAssertions, ...
        'scalar_assertions', scalarAssertions);
end

function count = assert_vector(actual, expected, description)
    if ~isequal(actual, expected)
        error('dtwc:f19WritebackOracle', ...
            '%s mismatch: actual=%s expected=%s.', ...
            description, mat2str(actual), mat2str(expected));
    end
    count = 1;
end

function count = assert_scalar(actual, expected, description)
    if ~isequal(actual, expected)
        error('dtwc:f19WritebackOracle', ...
            '%s mismatch: actual=%.17g expected=%.17g.', ...
            description, double(actual), double(expected));
    end
    count = 1;
end

function delete_problem(prob)
    if isvalid(prob)
        delete(prob);
    end
end
