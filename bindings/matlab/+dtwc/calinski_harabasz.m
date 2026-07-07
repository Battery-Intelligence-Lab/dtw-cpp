%> @file calinski_harabasz.m
%> @brief Calinski-Harabasz index (api-contract-2.0.md §2.4 canonical name).
%> @author Volkan Kumtepeli
function ch = calinski_harabasz(prob)
%CALINSKI_HARABASZ Compute the Calinski-Harabasz index (higher is better).
%
%   ch = dtwc.calinski_harabasz(prob)
%
%   Canonical 2.0 name (was dtwc.calinski_harabasz_index, kept as a deprecated
%   alias). Requires prior clustering stored in prob.
%
%   See also dtwc.silhouette, dtwc.davies_bouldin, dtwc.dunn
    ch = dtwc_mex('calinski_harabasz_index', prob.get_handle());
end
