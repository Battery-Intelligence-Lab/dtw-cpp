%> @file normalized_mutual_info.m
%> @brief Normalized mutual information (api-contract-2.0.md §2.4 canonical name).
%> @author Volkan Kumtepeli
function nmi = normalized_mutual_info(labels_true, labels_pred)
%NORMALIZED_MUTUAL_INFO Normalized mutual information between two labelings ([0,1]).
%
%   nmi = dtwc.normalized_mutual_info(labels_true, labels_pred)
%
%   Canonical 2.0 name (was dtwc.normalized_mutual_information, kept as a
%   deprecated alias). Both label vectors accept int32 or double, same length.
%
%   See also dtwc.adjusted_rand
    nmi = dtwc_mex('normalized_mutual_information', labels_true, labels_pred);
end
