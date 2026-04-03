function ari = adjusted_rand_index(labels1, labels2)
%ADJUSTED_RAND_INDEX Compute the Adjusted Rand Index between two clusterings.
%
%   ari = dtwc.adjusted_rand_index(labels1, labels2)
%
%   The ARI is a measure of agreement between two clusterings, adjusted
%   for chance. Values range from -1 to 1, with 1 meaning perfect match.
%
%   Parameters
%   ----------
%   labels1 : integer vector (1-based)
%       First set of cluster labels.
%   labels2 : integer vector (1-based)
%       Second set of cluster labels.
%
%   Returns
%   -------
%   ari : double scalar
%       The Adjusted Rand Index.
%
%   See also dtwc.normalized_mutual_information

    validateattributes(labels1, {'numeric', 'int32'}, {'vector', 'nonempty'}, 'adjusted_rand_index', 'labels1');
    validateattributes(labels2, {'numeric', 'int32'}, {'vector', 'nonempty'}, 'adjusted_rand_index', 'labels2');
    assert(numel(labels1) == numel(labels2), 'dtwc:sizeMismatch', 'Label vectors must have the same length.');

    ari = dtwc_mex('adjusted_rand_index', int32(labels1(:)'), int32(labels2(:)'));
end
