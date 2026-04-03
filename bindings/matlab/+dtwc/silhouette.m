%> @file silhouette.m
%> @brief Compute silhouette scores for each data point.
%> @author Volkan Kumtepeli
function s = silhouette(prob)
%SILHOUETTE Compute silhouette scores for each data point.
%
%   s = dtwc.silhouette(prob)
%
%   Requires a prior clustering (e.g., via dtwc.fast_pam). Silhouette
%   values range from -1 (misclassified) to +1 (well-clustered).
%
%   Parameters
%   ----------
%   prob : dtwc.Problem
%       Problem with clustering results stored (run fast_pam first).
%
%   Returns
%   -------
%   s : double row vector (1 x N)
%       Silhouette score for each data point.
%
%   See also dtwc.davies_bouldin_index, dtwc.fast_pam

    s = dtwc_mex('silhouette', prob.get_handle());
end
