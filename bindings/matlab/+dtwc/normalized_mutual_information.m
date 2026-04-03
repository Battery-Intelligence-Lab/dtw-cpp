%> @file normalized_mutual_information.m
%> @brief Compute NMI between two clusterings.
%> @author Volkan Kumtepeli
function nmi = normalized_mutual_information(labels1, labels2)
%NORMALIZED_MUTUAL_INFORMATION Compute NMI between two clusterings.
%
%   nmi = dtwc.normalized_mutual_information(labels1, labels2)
%
%   NMI measures the mutual information between two clusterings,
%   normalized to [0, 1]. A value of 1 means perfect agreement.
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
%   nmi : double scalar
%       The Normalized Mutual Information.
%
%   See also dtwc.adjusted_rand_index

    validateattributes(labels1, {'numeric', 'int32'}, {'vector', 'nonempty'}, 'normalized_mutual_information', 'labels1');
    validateattributes(labels2, {'numeric', 'int32'}, {'vector', 'nonempty'}, 'normalized_mutual_information', 'labels2');
    assert(numel(labels1) == numel(labels2), 'dtwc:sizeMismatch', 'Label vectors must have the same length.');

    nmi = dtwc_mex('normalized_mutual_information', int32(labels1(:)'), int32(labels2(:)'));
end
