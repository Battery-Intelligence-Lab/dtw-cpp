%> @file adtw_distance.m
%> @brief Compute Amerced DTW distance between two time series.
%> @author Volkan Kumtepeli
function d = adtw_distance(x, y, varargin)
%ADTW_DISTANCE Compute Amerced DTW distance between two time series.
%
%   d = dtwc.adtw_distance(x, y)
%   d = dtwc.adtw_distance(x, y, 'Band', 10, 'Penalty', 2.0)
%
%   ADTW adds a penalty for non-diagonal (horizontal/vertical) warping
%   steps, discouraging time stretching/compression.
%
%   Parameters
%   ----------
%   x : numeric vector
%       First time series.
%   y : numeric vector
%       Second time series.
%   Band : int, optional (default -1)
%       Sakoe-Chiba band width. Use -1 for full DTW.
%   Penalty : double, optional (default 1.0)
%       Penalty for non-diagonal warping steps.
%
%   Returns
%   -------
%   d : double
%       The ADTW distance between x and y.
%
%   Reference: Herrmann & Shifaz (2023), "Amercing: An intuitive and
%              effective constraint for dynamic time warping"
%
%   See also dtwc.dtw_distance, dtwc.wdtw_distance

    p = inputParser;
    addRequired(p, 'x', @isnumeric);
    addRequired(p, 'y', @isnumeric);
    addParameter(p, 'Band', -1, @(v) isnumeric(v) && isscalar(v));
    addParameter(p, 'Penalty', 1.0, @(v) isnumeric(v) && isscalar(v));
    parse(p, x, y, varargin{:});

    d = dtwc_mex('adtw_distance', ...
                  double(x(:)'), ...
                  double(y(:)'), ...
                  double(p.Results.Band), ...
                  double(p.Results.Penalty));
end
