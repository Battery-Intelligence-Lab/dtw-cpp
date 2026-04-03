function val = inertia(prob)
%INERTIA Compute the inertia (total within-cluster distance sum).
%
%   val = dtwc.inertia(prob)
%
%   Sum of distances from each point to its nearest medoid.
%   Requires prior clustering.
%
%   Parameters
%   ----------
%   prob : dtwc.Problem
%       Problem with clustering results stored.
%
%   Returns
%   -------
%   val : double scalar
%       The inertia value.
%
%   See also dtwc.silhouette, dtwc.calinski_harabasz_index

    val = dtwc_mex('inertia', prob.get_handle());
end
