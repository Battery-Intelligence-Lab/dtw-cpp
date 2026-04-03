%> @file fast_pam.m
%> @brief Run FastPAM1 k-medoids clustering on a Problem.
%> @author Volkan Kumtepeli
function result = fast_pam(prob, k, varargin)
%FAST_PAM Run FastPAM1 k-medoids clustering on a Problem.
%
%   result = dtwc.fast_pam(prob, k)
%   result = dtwc.fast_pam(prob, k, 'MaxIter', 200)
%
%   FastPAM1 considers swapping any medoid with any non-medoid globally
%   (true PAM SWAP), achieving the same quality as PAM with O(k) speedup.
%
%   Parameters
%   ----------
%   prob : dtwc.Problem
%       Problem with data loaded. Distance matrix filled automatically.
%   k : positive integer
%       Number of clusters.
%   MaxIter : int, optional (default 100)
%       Maximum SWAP iterations.
%
%   Returns
%   -------
%   result : struct with fields:
%       labels          - int32 row vector (1-based cluster assignments)
%       medoid_indices  - int32 row vector (1-based medoid indices)
%       total_cost      - double scalar
%       iterations      - int32 scalar
%       converged       - logical scalar
%
%   Note: Results are also stored back into prob for use by scoring
%   functions (e.g., dtwc.silhouette(prob)).
%
%   Reference: Schubert & Rousseeuw (2021), "Fast and eager k-medoids"
%
%   See also dtwc.fast_clara, dtwc.clarans, dtwc.silhouette

    p = inputParser;
    addRequired(p, 'prob');
    addRequired(p, 'k', @(v) isnumeric(v) && isscalar(v) && v > 0);
    addParameter(p, 'MaxIter', 100, @(v) isnumeric(v) && isscalar(v) && v > 0);
    parse(p, prob, k, varargin{:});

    result = dtwc_mex('fast_pam', prob.get_handle(), double(k), double(p.Results.MaxIter));
end
