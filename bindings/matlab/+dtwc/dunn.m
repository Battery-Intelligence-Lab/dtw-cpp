function di = dunn(prob)
%DUNN Compute the Dunn index (higher is better).
%
%   di = dtwc.dunn(prob)
%
%   Requires prior clustering stored in prob.
%
%   See also dtwc.silhouette, dtwc.davies_bouldin

%> @file dunn.m
%> @brief Dunn index.
%> @author Volkan Kumtepeli
%> @date 07 Jul 2026
    di = dtwc_mex('dunn', prob.get_handle());
end
