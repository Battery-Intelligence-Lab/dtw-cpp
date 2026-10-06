%> @file calinski_harabasz.m
%> @brief Calinski-Harabasz index.
%> @author Volkan Kumtepeli
function ch = calinski_harabasz(prob)
%CALINSKI_HARABASZ Compute the Calinski-Harabasz index (higher is better).
%
%   ch = dtwc.calinski_harabasz(prob)
%
%   Requires prior clustering stored in prob.
%
%   See also dtwc.silhouette, dtwc.davies_bouldin, dtwc.dunn
    ch = dtwc_mex('calinski_harabasz', prob.get_handle());
end
