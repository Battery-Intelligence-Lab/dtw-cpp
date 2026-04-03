%> @file cut_dendrogram.m
%> @brief Cut a dendrogram to produce k flat clusters.
%> @author Volkan Kumtepeli
function result = cut_dendrogram(dend, prob, k)
%CUT_DENDROGRAM Cut a dendrogram to produce k flat clusters.
%
%   result = dtwc.cut_dendrogram(dend, prob, k)
%
%   Replays the last N-k merges using union-find, assigns medoids by
%   minimizing intra-cluster distance sums.
%
%   Parameters
%   ----------
%   dend : struct
%       Dendrogram returned by dtwc.build_dendrogram.
%   prob : dtwc.Problem
%       Problem whose distance matrix is used for medoid computation.
%   k : positive integer
%       Number of clusters (1 <= k <= dend.n_points).
%
%   Returns
%   -------
%   result : struct (same fields as dtwc.fast_pam)
%
%   See also dtwc.build_dendrogram, dtwc.fast_pam
% @author Volkan Kumtepeli

    validateattributes(k, {'numeric'}, {'scalar', 'positive', 'integer'}, 'cut_dendrogram', 'k');
    result = dtwc_mex('cut_dendrogram', dend, prob.get_handle(), double(k));
end
