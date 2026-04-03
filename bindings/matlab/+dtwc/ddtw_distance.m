%> @file ddtw_distance.m
%> @brief Compute Derivative DTW distance between two time series.
%> @author Volkan Kumtepeli
function d = ddtw_distance(x, y, varargin)
%DDTW_DISTANCE Compute Derivative DTW distance between two time series.
%
%   d = dtwc.ddtw_distance(x, y)
%   d = dtwc.ddtw_distance(x, y, 'Band', 10)
%
%   DDTW applies a derivative transform to both series, then computes
%   standard DTW on the transformed series. This captures shape similarity
%   rather than amplitude similarity.
%
%   Parameters
%   ----------
%   x : numeric vector
%       First time series.
%   y : numeric vector
%       Second time series.
%   Band : int, optional (default -1)
%       Sakoe-Chiba band width. Use -1 for full (unconstrained) DTW.
%
%   Returns
%   -------
%   d : double
%       The DDTW distance between x and y.
%
%   Reference: Keogh & Pazzani (2001), "Derivative Dynamic Time Warping"
%
%   See also dtwc.dtw_distance, dtwc.derivative_transform

    p = inputParser;
    addRequired(p, 'x', @isnumeric);
    addRequired(p, 'y', @isnumeric);
    addParameter(p, 'Band', -1, @(v) isnumeric(v) && isscalar(v));
    parse(p, x, y, varargin{:});

    d = dtwc_mex('ddtw_distance', ...
                  double(x(:)'), ...
                  double(y(:)'), ...
                  double(p.Results.Band));
end
