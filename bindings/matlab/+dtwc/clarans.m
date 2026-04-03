function result = clarans(prob, k, varargin)
%CLARANS Run CLARANS: randomized k-medoids via neighborhood search.
%
%   result = dtwc.clarans(prob, k)
%   result = dtwc.clarans(prob, k, 'NumLocal', 5, 'MaxNeighbor', 500)
%
%   CLARANS iteratively tests random (medoid_out, x_in) swaps, accepting
%   only strictly improving ones. Best result across restarts is returned.
%
%   Parameters
%   ----------
%   prob : dtwc.Problem
%       Problem with data loaded.
%   k : positive integer
%       Number of clusters.
%   NumLocal : int, optional (default 2)
%       Number of random restarts.
%   MaxNeighbor : int, optional (default -1 = auto)
%       Max non-improving swaps per restart.
%   MaxDtwEvals : int, optional (default -1 = no limit)
%       Hard budget on total DTW computations.
%   Seed : int, optional (default 42)
%       Random seed for reproducibility.
%
%   Returns
%   -------
%   result : struct (same fields as dtwc.fast_pam)
%
%   Reference: Ng & Han (2002), "CLARANS: A method for clustering objects"
%
%   See also dtwc.fast_pam, dtwc.fast_clara

    p = inputParser;
    addRequired(p, 'prob');
    addRequired(p, 'k', @(v) isnumeric(v) && isscalar(v) && v > 0);
    addParameter(p, 'NumLocal', 2, @(v) isnumeric(v) && isscalar(v) && v > 0);
    addParameter(p, 'MaxNeighbor', -1, @(v) isnumeric(v) && isscalar(v));
    addParameter(p, 'MaxDtwEvals', -1, @(v) isnumeric(v) && isscalar(v));
    addParameter(p, 'Seed', 42, @(v) isnumeric(v) && isscalar(v));
    parse(p, prob, k, varargin{:});

    result = dtwc_mex('clarans', prob.get_handle(), ...
        double(k), ...
        double(p.Results.NumLocal), ...
        double(p.Results.MaxNeighbor), ...
        double(p.Results.MaxDtwEvals), ...
        double(p.Results.Seed));
end
