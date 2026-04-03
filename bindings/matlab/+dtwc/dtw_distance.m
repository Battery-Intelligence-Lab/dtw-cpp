%> @file dtw_distance.m
%> @brief Compute DTW distance between two time series using DTWC++.
%> @author Volkan Kumtepeli
function d = dtw_distance(x, y, varargin)
%DTW_DISTANCE Compute DTW distance between two time series using DTWC++.
%
%   d = dtwc.dtw_distance(x, y)
%   d = dtwc.dtw_distance(x, y, 'Band', 10)
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
%       The DTW distance between x and y.
%
%   Examples
%   --------
%       x = [1 2 3 4 5];
%       y = [2 4 6 3 1];
%       d = dtwc.dtw_distance(x, y);
%       d_banded = dtwc.dtw_distance(x, y, 'Band', 2);
%
%   See also dtwc.ddtw_distance, dtwc.compute_distance_matrix, dtwc.DTWClustering
% @author Volkan Kumtepeli

    p = inputParser;
    addRequired(p, 'x', @isnumeric);
    addRequired(p, 'y', @isnumeric);
    addParameter(p, 'Band', -1, @(v) isnumeric(v) && isscalar(v));
    parse(p, x, y, varargin{:});

    d = dtwc_mex('dtw_distance', ...
                  double(x(:)'), ...
                  double(y(:)'), ...
                  double(p.Results.Band));
end
