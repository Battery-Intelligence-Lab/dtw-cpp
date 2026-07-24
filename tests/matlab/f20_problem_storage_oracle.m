function summary = f20_problem_storage_oracle( ...
    repoRoot, mexPath, expectedRelease, profile, scratch)
%F20_PROBLEM_STORAGE_ORACLE Real MATLAB binding gate for Problem storage.
%   Drives set_storage_policy -> set_data through a fresh, path-identified MEX.
%   PROFILE is "llfio-on" or "llfio-off"; neither route may skip.

repoRoot = char(repoRoot);
mexPath = char(mexPath);
expectedRelease = char(expectedRelease);
profile = char(profile);
scratch = char(scratch);

must(any(strcmp(profile, {'llfio-on', 'llfio-off'})), ...
    ['unknown profile: ' profile]);
must(exist(repoRoot, 'dir') == 7, ['repository root is absent: ' repoRoot]);
must(exist(mexPath, 'file') ~= 0, ['MEX is absent: ' mexPath]);
must(exist(scratch, 'dir') == 7, ['scratch directory is absent: ' scratch]);

buildPrefix = [normalise_path(fullfile(repoRoot, 'build')) '/'];
scratchPrefix = [normalise_path(scratch) '/'];
must(startsWith(scratchPrefix, buildPrefix), ...
    ['scratch directory is outside the repository build root: ' scratch]);

observedRelease = ['R' char(version('-release'))];
must(strcmp(observedRelease, expectedRelease), ...
    sprintf('release mismatch: expected=%s observed=%s', ...
        expectedRelease, observedRelease));

restoredefaultpath;
cd(repoRoot);
addpath(fullfile(repoRoot, 'bindings', 'matlab'));
addpath(fullfile(repoRoot, 'tests', 'matlab'));
addpath(fileparts(mexPath)); % Last addpath prepends the fresh MEX directory.
clear dtwc_mex;
rehash;

mexPaths = which('dtwc_mex', '-all');
if ischar(mexPaths)
    if isempty(mexPaths)
        mexPaths = {};
    else
        mexPaths = {mexPaths};
    end
elseif isstring(mexPaths)
    mexPaths = cellstr(mexPaths);
end
must(numel(mexPaths) == 1, ...
    sprintf('expected one dtwc_mex path, observed %d', numel(mexPaths)));
must(strcmp(normalise_path(mexPaths{1}), normalise_path(mexPath)), ...
    sprintf('resolved MEX mismatch: expected=%s observed=%s', ...
        mexPath, mexPaths{1}));
fprintf(['F20_MATLAB_MEX requested_release=%s observed_release=%s ' ...
    'build=%s path=%s\n'], ...
    expectedRelease, observedRelease, profile, mexPaths{1});

must(isempty(dir(fullfile(scratch, '*.dtws'))), ...
    'scratch directory already contains a .dtws artifact');

X = [ ...
    0,       115.522, 52.056, 61.269, 60.104, 96.325; ...
    115.522, 0,       86.568, 64.249, 63.708, 80.603; ...
    52.056,  86.568,  0,      59.536, 85.575, 93.885; ...
    61.269,  64.249,  59.536, 0,      28.114, 81.654; ...
    60.104,  63.708,  85.575, 28.114, 0,      69.942; ...
    96.325,  80.603,  93.885, 81.654, 69.942, 0];
names = {'1', '2', '3', '4', '5', '6'};
ndim = 2;
must(numel(X) * 8 == 288, 'fixture footprint is not 288 bytes');
oracle = independent_oracle(X, ndim);

heap = dtwc.Problem('f20_matlab_heap');
heapCleanup = onCleanup(@() safe_delete(heap));
heap.set_storage_policy('heap');
heap.set_data(X, names, ndim);
fprintf('F20_MATLAB_STAGE build=%s stage=heap-set-data\n', profile);
must(heap.size() == 6, 'Heap set_data did not publish six series');
must(isempty(dir(fullfile(scratch, '*.dtws'))), ...
    'forced Heap unexpectedly created a .dtws artifact');
heapDistances = verify_distances(heap, oracle);
fprintf('F20_MATLAB_STAGE build=%s stage=heap-distances\n', profile);
heapResult = dtwc.fast_pam(heap, 2, 'MaxIter', 100, 'Seed', 42);
fprintf('F20_MATLAB_STAGE build=%s stage=heap-fastpam\n', profile);

candidate = dtwc.Problem('f20_matlab_candidate');
candidateCleanup = onCleanup(@() safe_delete(candidate));
sentinel = [-7, -3, 11, 2; 5, 13, -2, 17];
sentinelNames = {'sentinel-a', 'sentinel-b'};
sentinelOracle = dependent_l1_dtw(sentinel(1, :), sentinel(2, :), ndim);
candidate.set_storage_policy('heap');
candidate.set_data(sentinel, sentinelNames, ndim);
candidate.set_distance_matrix([0, 123; 123, 0]);
candidate.set_storage_policy('mmap');
fprintf('F20_MATLAB_STAGE build=%s stage=sentinel-ready\n', profile);
must(candidate.size() == 2, ...
    'policy change retroactively replaced sentinel series');
must(candidate.is_distance_matrix_filled(), ...
    'policy change retroactively invalidated sentinel matrix');
must(isequal(candidate.distance_matrix(), [0, 123; 123, 0]), ...
    'policy change retroactively changed sentinel matrix');

if strcmp(profile, 'llfio-on')
    fprintf('F20_MATLAB_STAGE build=%s stage=mmap-set-data-enter\n', profile);
    candidate.set_data(X, names, ndim);
    fprintf('F20_MATLAB_STAGE build=%s stage=mmap-set-data-exit\n', profile);
    must(candidate.size() == 6, 'Mmap set_data did not publish six series');
    artifacts = dir(fullfile(scratch, '*.dtws'));
    must(numel(artifacts) == 1, ...
        sprintf('expected one Mmap artifact, observed %d', numel(artifacts)));
    mappedDistances = verify_distances(candidate, oracle);
    fprintf('F20_MATLAB_STAGE build=%s stage=mmap-distances\n', profile);
    mappedResult = dtwc.fast_pam( ...
        candidate, 2, 'MaxIter', 100, 'Seed', 42);
    fprintf('F20_MATLAB_STAGE build=%s stage=mmap-fastpam\n', profile);
    verify_fastpam_equal(heapResult, mappedResult);

    delete(candidate);
    clear candidate;
    clear candidateCleanup;
    artifactPath = fullfile(artifacts(1).folder, artifacts(1).name);
    verify_store_artifact(artifactPath, X);

    delete(heap);
    clear heap;
    clear heapCleanup;
    summary = struct( ...
        'release', observedRelease, ...
        'profile', profile, ...
        'distances', heapDistances + mappedDistances, ...
        'artifacts', 1, ...
        'skips', 0, ...
        'verdict', 'PASS');
    fprintf(['F20_MATLAB_STORAGE requested_release=%s ' ...
        'observed_release=%s build=llfio-on route=mmap ' ...
        'artifact=408/408 distances=72/72 fastpam=exact ' ...
        'transaction=pass subject_skips=0 verdict=PASS\n'], ...
        expectedRelease, observedRelease);
else
    expectedError = [ ...
        'Problem::set_data: StoragePolicy::Mmap requested but mmap ' ...
        'support (llfio) is not compiled in. Rebuild with ' ...
        '-DDTWC_ENABLE_LLFIO=ON.'];
    caught = false;
    try
        fprintf('F20_MATLAB_STAGE build=%s stage=mmap-reject-enter\n', profile);
        candidate.set_data(X, names, ndim);
    catch exception
        caught = true;
        must(strcmp(exception.identifier, 'dtwc:ioError'), ...
            ['wrong Mmap error identifier: ' exception.identifier]);
        must(strcmp(exception.message, expectedError), ...
            sprintf('wrong Mmap error message:\n%s', exception.message));
    end
    must(caught, 'explicit Mmap unexpectedly succeeded without llfio');
    must(candidate.size() == 2, ...
        'failed Mmap set_data changed sentinel series count');
    must(candidate.is_distance_matrix_filled(), ...
        'failed Mmap set_data invalidated sentinel matrix');
    must(isequal(candidate.distance_matrix(), [0, 123; 123, 0]), ...
        'failed Mmap set_data changed sentinel matrix');
    candidate.refresh_distance_matrix();
    observedSentinel = candidate.dist_by_ind(1, 2);
    must(same_bits(observedSentinel, sentinelOracle), ...
        sprintf('failed Mmap set_data changed sentinel bytes: %s vs %s', ...
            num2hex(observedSentinel), num2hex(sentinelOracle)));
    must(isempty(dir(fullfile(scratch, '*.dtws'))), ...
        'llfio-OFF rejection left a .dtws artifact');

    delete(candidate);
    clear candidate;
    clear candidateCleanup;
    delete(heap);
    clear heap;
    clear heapCleanup;
    summary = struct( ...
        'release', observedRelease, ...
        'profile', profile, ...
        'distances', heapDistances, ...
        'artifacts', 0, ...
        'skips', 0, ...
        'verdict', 'PASS');
    fprintf(['F20_MATLAB_STORAGE requested_release=%s ' ...
        'observed_release=%s build=llfio-off route=rejected ' ...
        'error_id=dtwc:ioError transaction=pass artifacts=0 ' ...
        'distances=36/36 subject_skips=0 verdict=PASS\n'], ...
        expectedRelease, observedRelease);
end
end

function oracle = independent_oracle(X, ndim)
expectedUpperHex = { ...
    '4071fdcac083126f', '406456b851eb851f', '406c7e76c8b43958', ...
    '4070914395810625', '4072b9ef9db22d0f', '4071476c8b439581', ...
    '406ecdb22d0e5604', '406cd3ef9db22d0e', '406a6b7ced916872', ...
    '406b8978d4fdf3b7', '406e59c28f5c28f6', '40713c147ae147af', ...
    '4057ebd70a3d70a4', '406d6a353f7ced91', '406fdb3333333332'};
n = size(X, 1);
must(n == 6 && size(X, 2) == 6, 'fixture is not 6-by-6');
oracle = zeros(n, n);
upper = 0;
for i = 1:n
    for j = (i + 1):n
        upper = upper + 1;
        value = dependent_l1_dtw(X(i, :), X(j, :), ndim);
        must(strcmpi(num2hex(value), expectedUpperHex{upper}), ...
            sprintf('independent DP bit mismatch at pair %d,%d: %s', ...
                i, j, num2hex(value)));
        oracle(i, j) = value;
        oracle(j, i) = value;
    end
end
must(upper == 15, 'independent DP did not execute 15 upper pairs');
end

function value = dependent_l1_dtw(lhs, rhs, ndim)
must(mod(numel(lhs), ndim) == 0, 'lhs length is not divisible by ndim');
must(mod(numel(rhs), ndim) == 0, 'rhs length is not divisible by ndim');
n = numel(lhs) / ndim;
m = numel(rhs) / ndim;
dp = inf(n + 1, m + 1);
dp(1, 1) = 0;
for i = 1:n
    for j = 1:m
        lhsBase = (i - 1) * ndim;
        rhsBase = (j - 1) * ndim;
        local = 0;
        for d = 1:ndim
            local = local + abs(lhs(lhsBase + d) - rhs(rhsBase + d));
        end
        dp(i + 1, j + 1) = local + min([ ...
            dp(i, j + 1), dp(i + 1, j), dp(i, j)]);
    end
end
value = dp(n + 1, m + 1);
end

function count = verify_distances(problem, oracle)
count = 0;
for i = 1:size(oracle, 1)
    for j = 1:size(oracle, 2)
        observed = problem.dist_by_ind(i, j);
        expected = oracle(i, j);
        must(same_bits(observed, expected), ...
            sprintf('distance bit mismatch at %d,%d: %s vs %s', ...
                i, j, num2hex(observed), num2hex(expected)));
        count = count + 1;
    end
end
must(count == 36, 'public Problem route did not execute 36 distances');
end

function verify_fastpam_equal(heapResult, mappedResult)
must(isequal(heapResult.labels, mappedResult.labels), ...
    'Heap/Mmap FastPAM labels differ');
must(isequal(heapResult.medoid_indices, mappedResult.medoid_indices), ...
    'Heap/Mmap FastPAM medoids differ');
must(same_bits(heapResult.total_cost, mappedResult.total_cost), ...
    'Heap/Mmap FastPAM total costs differ');
must(isequal(heapResult.iterations, mappedResult.iterations), ...
    'Heap/Mmap FastPAM iteration counts differ');
must(isequal(heapResult.converged, mappedResult.converged), ...
    'Heap/Mmap FastPAM convergence flags differ');
end

function verify_store_artifact(path, X)
[~, ~, byteOrder] = computer;
must(byteOrder == 'L', ...
    'F20 .dtws byte oracle requires a little-endian host');
file = fopen(path, 'rb');
must(file ~= -1, ['cannot open Mmap artifact: ' path]);
fileCleanup = onCleanup(@() fclose(file));
[column, count] = fread(file, inf, '*uint8');
clear fileCleanup;
bytes = reshape(column, 1, []);
must(count == 408 && numel(bytes) == 408, ...
    sprintf('Mmap artifact size mismatch: %d', count));

header = zeros(1, 64, 'uint8');
header(1:4) = uint8('DTWS');
header(5:6) = uint8([1, 0]);
header(7:10) = uint8([4, 3, 2, 1]);
header(11) = uint8(8);
header(13) = uint8(6);
header(21) = uint8(2);
header(29:32) = uint8([228, 160, 129, 171]); % 0xab81a0e4, little endian
offsetBytes = typecast(uint64([0, 48, 96, 144, 192, 240, 288]), 'uint8');
payload = reshape(typecast(reshape(X.', 1, []), 'uint8'), 1, []);
expected = [header, reshape(offsetBytes, 1, []), payload];
must(numel(expected) == 408, 'independent .dtws byte oracle is not 408 bytes');
must(isequal(bytes, expected), 'Mmap artifact differs from exact .dtws oracle');
end

function result = same_bits(lhs, rhs)
result = strcmpi(num2hex(double(lhs)), num2hex(double(rhs)));
end

function path = normalise_path(path)
path = lower(strrep(char(path), '\', '/'));
path = regexprep(path, '/+$', '');
end

function safe_delete(object)
try
    delete(object);
catch
end
end

function must(condition, message)
if ~condition
    error('dtwc:f20MatlabStorageGate', '%s', message);
end
end
