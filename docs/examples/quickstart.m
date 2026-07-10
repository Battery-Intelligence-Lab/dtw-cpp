% Executable MATLAB version of the documentation quickstart.
csv = 'tests/conformance/data/conformance_series.csv';
if exist('DTWC_QUICKSTART_CSV', 'var'); csv = DTWC_QUICKSTART_CSV; end

dtwc.device('cpu');
data = dtwc.load(csv, 'delimiter', ',', 'name', 'quickstart');
result = dtwc.cluster(data, 3, 'method', 'pam', 'band', 3, ...
    'device', 'cpu', 'max_iter', 100);

medoids = sort(double(result.medoids(:)') - 1); % canonical 0-based output
rawMedoids = double(result.medoids(:)') - 1;
rawLabels = double(result.labels(:)');          % 1-based index into medoids
assigned = rawMedoids(rawLabels);
labels = arrayfun(@(m) find(medoids == m, 1) - 1, assigned);
expected = [zeros(1, 9), ones(1, 9), 2 * ones(1, 9)];
assert(isequal(labels, expected) && isequal(medoids, [4, 13, 22]));

fprintf('labels: 0x9 1x9 2x9\n');
fprintf('medoids: 4 13 22\n');
fprintf('mean silhouette: %.6f\n', result.score('silhouette'));
