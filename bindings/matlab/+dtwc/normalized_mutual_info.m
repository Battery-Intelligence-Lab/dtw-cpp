%> @file normalized_mutual_info.m
%> @brief Normalized mutual information (api-contract-2.0.md §2.4 canonical name).
%> @author Volkan Kumtepeli
function nmi = normalized_mutual_info(labels_true, labels_pred)
%NORMALIZED_MUTUAL_INFO Normalized mutual information between two labelings ([0,1]).
%
%   nmi = dtwc.normalized_mutual_info(labels_true, labels_pred)
%
%   Both label vectors accept int32 or double and must be the same length. A
%   label is a value, not an index: any integer is one, 0 and negatives
%   included. NaN, Inf or a fractional value raises dtwc:invalidArgument.
%
%   See also dtwc.adjusted_rand
    nmi = dtwc_mex('normalized_mutual_info', labels_true, labels_pred);
end
