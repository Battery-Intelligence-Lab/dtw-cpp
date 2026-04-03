function d = soft_dtw_distance(x, y, varargin)
%SOFT_DTW_DISTANCE Compute Soft-DTW distance between two time series.
%
%   d = dtwc.soft_dtw_distance(x, y)
%   d = dtwc.soft_dtw_distance(x, y, 'Gamma', 0.1)
%
%   Soft-DTW replaces the hard min in DTW with a differentiable softmin.
%   As gamma -> 0, Soft-DTW converges to standard DTW.
%   Note: Soft-DTW can be NEGATIVE for identical series when gamma > 0.
%
%   Parameters
%   ----------
%   x : numeric vector
%       First time series.
%   y : numeric vector
%       Second time series.
%   Gamma : double, optional (default 1.0)
%       Smoothing parameter. Lower = closer to hard DTW.
%
%   Returns
%   -------
%   d : double
%       The Soft-DTW distance between x and y.
%
%   Reference: Cuturi & Blondel (2017), "Soft-DTW: a Differentiable Loss
%              Function for Time-Series"
%
%   See also dtwc.soft_dtw_gradient, dtwc.dtw_distance

    p = inputParser;
    addRequired(p, 'x', @isnumeric);
    addRequired(p, 'y', @isnumeric);
    addParameter(p, 'Gamma', 1.0, @(v) isnumeric(v) && isscalar(v) && v > 0);
    parse(p, x, y, varargin{:});

    d = dtwc_mex('soft_dtw_distance', ...
                  double(x(:)'), ...
                  double(y(:)'), ...
                  double(p.Results.Gamma));
end
