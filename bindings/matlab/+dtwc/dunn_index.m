function di = dunn_index(prob)
%DUNN_INDEX Compute the Dunn index.
%
%   di = dtwc.dunn_index(prob)
%
%   Higher values indicate better clustering (compact, well-separated).
%   Requires prior clustering.
%
%   Parameters
%   ----------
%   prob : dtwc.Problem
%       Problem with clustering results stored.
%
%   Returns
%   -------
%   di : double scalar
%       The Dunn index.
%
%   See also dtwc.silhouette, dtwc.davies_bouldin_index

    di = dtwc_mex('dunn_index', prob.get_handle());
end
