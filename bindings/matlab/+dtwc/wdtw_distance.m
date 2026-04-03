%> @file wdtw_distance.m
%> @brief Compute Weighted DTW distance between two time series.
%> @author Volkan Kumtepeli
function d = wdtw_distance(x, y, varargin)
%WDTW_DISTANCE Compute Weighted DTW distance between two time series.
%
%   d = dtwc.wdtw_distance(x, y)
%   d = dtwc.wdtw_distance(x, y, 'Band', 10, 'G', 0.1)
%
%   WDTW multiplies each pointwise distance by a weight depending on the
%   index difference |i - j|, penalizing large time shifts.
%
%   Parameters
%   ----------
%   x : numeric vector
%       First time series.
%   y : numeric vector
%       Second time series.
%   Band : int, optional (default -1)
%       Sakoe-Chiba band width. Use -1 for full DTW.
%   G : double, optional (default 0.05)
%       Steepness of the logistic weight function.
%
%   Returns
%   -------
%   d : double
%       The WDTW distance between x and y.
%
%   Reference: Jeong et al. (2011), "Weighted dynamic time warping"
%
%   See also dtwc.dtw_distance, dtwc.adtw_distance
% @author Volkan Kumtepeli

    p = inputParser;
    addRequired(p, 'x', @isnumeric);
    addRequired(p, 'y', @isnumeric);
    addParameter(p, 'Band', -1, @(v) isnumeric(v) && isscalar(v));
    addParameter(p, 'G', 0.05, @(v) isnumeric(v) && isscalar(v));
    parse(p, x, y, varargin{:});

    d = dtwc_mex('wdtw_distance', ...
                  double(x(:)'), ...
                  double(y(:)'), ...
                  double(p.Results.Band), ...
                  double(p.Results.G));
end
