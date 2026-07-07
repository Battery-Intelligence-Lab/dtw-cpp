%> @file davies_bouldin.m
%> @brief Davies-Bouldin index (api-contract-2.0.md §2.4 canonical name).
%> @author Volkan Kumtepeli
function db = davies_bouldin(prob)
%DAVIES_BOULDIN Compute the Davies-Bouldin index (lower is better).
%
%   db = dtwc.davies_bouldin(prob)
%
%   Canonical 2.0 name (was dtwc.davies_bouldin_index, kept as a deprecated alias).
%   Requires prior clustering stored in prob.
%
%   See also dtwc.silhouette, dtwc.dunn, dtwc.calinski_harabasz
    db = dtwc_mex('davies_bouldin_index', prob.get_handle());
end
