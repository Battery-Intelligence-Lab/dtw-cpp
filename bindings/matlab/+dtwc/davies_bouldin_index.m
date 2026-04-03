%> @file davies_bouldin_index.m
%> @brief Compute the Davies-Bouldin index.
%> @author Volkan Kumtepeli
function db = davies_bouldin_index(prob)
%DAVIES_BOULDIN_INDEX Compute the Davies-Bouldin index.
%
%   db = dtwc.davies_bouldin_index(prob)
%
%   Lower values indicate better clustering. Requires prior clustering.
%
%   Parameters
%   ----------
%   prob : dtwc.Problem
%       Problem with clustering results stored.
%
%   Returns
%   -------
%   db : double scalar
%       The Davies-Bouldin index.
%
%   See also dtwc.silhouette, dtwc.dunn_index
% @author Volkan Kumtepeli

    db = dtwc_mex('davies_bouldin_index', prob.get_handle());
end
