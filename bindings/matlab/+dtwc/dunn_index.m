%> @file dunn_index.m
%> @brief Compute the Dunn index.
%> @author Volkan Kumtepeli
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
%   See also dtwc.silhouette, dtwc.davies_bouldin

    warning('dtwc:deprecatedAlias', ...
        ['''dtwc.dunn_index'' is deprecated; use ' ...
         '''dtwc.dunn'' instead.']);
    di = dtwc.dunn(prob);
end
