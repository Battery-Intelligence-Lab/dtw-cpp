function g = soft_dtw_gradient(x, y, varargin)
%SOFT_DTW_GRADIENT Compute gradient of Soft-DTW w.r.t. first series.
%
%   g = dtwc.soft_dtw_gradient(x, y)
%   g = dtwc.soft_dtw_gradient(x, y, 'Gamma', 0.1)
%
%   Returns the gradient of the Soft-DTW loss with respect to x, useful
%   for gradient-based optimization (e.g., DBA averaging with Soft-DTW).
%
%   Parameters
%   ----------
%   x : numeric vector
%       First time series (gradient is w.r.t. this).
%   y : numeric vector
%       Second time series.
%   Gamma : double, optional (default 1.0)
%       Smoothing parameter.
%
%   Returns
%   -------
%   g : double row vector
%       Gradient vector of size length(x).
%
%   Reference: Cuturi & Blondel (2017), "Soft-DTW: a Differentiable Loss
%              Function for Time-Series"
%
%   See also dtwc.distance.dtw

%> @file soft_dtw_gradient.m
%> @brief Compute gradient of Soft-DTW w.r.t. first series.
%> @author Volkan Kumtepeli
%> @date 03 Apr 2026
    is_vec = @(v) isnumeric(v) && ~isempty(v) && isvector(v);
    p = inputParser;
    addRequired(p, 'x', is_vec);
    addRequired(p, 'y', is_vec);
    addParameter(p, 'Gamma', 1.0, @(v) isnumeric(v) && isscalar(v) && v > 0);
    parse(p, x, y, varargin{:});

    g = dtwc_mex('soft_dtw_gradient', ...
                  double(x(:)'), ...
                  double(y(:)'), ...
                  double(p.Results.Gamma));
end
