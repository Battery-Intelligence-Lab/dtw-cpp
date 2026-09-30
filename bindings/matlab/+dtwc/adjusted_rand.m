%> @file adjusted_rand.m
%> @brief Adjusted Rand index (api-contract-2.0.md §2.4 canonical name).
%> @author Volkan Kumtepeli
function ari = adjusted_rand(labels_true, labels_pred)
%ADJUSTED_RAND Adjusted Rand index between two labelings (in [-1, 1]).
%
%   ari = dtwc.adjusted_rand(labels_true, labels_pred)
%
%   Both label vectors accept int32 or double and must be the same length. A
%   label is a value, not an index: any integer is one, 0 and negatives
%   included. NaN, Inf or a fractional value raises dtwc:invalidArgument.
%
%   See also dtwc.normalized_mutual_info
    ari = dtwc_mex('adjusted_rand', labels_true, labels_pred);
end
