%> @file calinski_harabasz_index.m
%> @brief Compute the Calinski-Harabasz index.
%> @author Volkan Kumtepeli
function ch = calinski_harabasz_index(prob)
%CALINSKI_HARABASZ_INDEX Compute the Calinski-Harabasz index.
%
%   ch = dtwc.calinski_harabasz_index(prob)
%
%   Higher values indicate better-defined clusters (higher between-cluster
%   dispersion relative to within-cluster dispersion).
%   Requires prior clustering.
%
%   Parameters
%   ----------
%   prob : dtwc.Problem
%       Problem with clustering results stored.
%
%   Returns
%   -------
%   ch : double scalar
%       The Calinski-Harabasz index.
%
%   See also dtwc.silhouette, dtwc.davies_bouldin_index
% @author Volkan Kumtepeli

    ch = dtwc_mex('calinski_harabasz_index', prob.get_handle());
end
