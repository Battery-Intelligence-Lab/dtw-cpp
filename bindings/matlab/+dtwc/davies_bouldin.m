%> @file davies_bouldin.m
%> @brief Davies-Bouldin index.
%> @author Volkan Kumtepeli
function db = davies_bouldin(prob)
%DAVIES_BOULDIN Compute the Davies-Bouldin index (lower is better).
%
%   db = dtwc.davies_bouldin(prob)
%
%   Requires prior clustering stored in prob.
%
%   See also dtwc.silhouette, dtwc.dunn, dtwc.calinski_harabasz
    db = dtwc_mex('davies_bouldin', prob.get_handle());
end
