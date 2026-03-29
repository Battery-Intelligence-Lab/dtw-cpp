function d = dtw_distance(x, y, varargin)
%DTW_DISTANCE Compute DTW distance between two time series using DTWC++.
%
%   d = dtwc.dtw_distance(x, y)
%   d = dtwc.dtw_distance(x, y, 'Band', 10)
%   d = dtwc.dtw_distance(x, y, 'Metric', 'l1')
%
%   Parameters
%   ----------
%   x : numeric vector
%       First time series.
%   y : numeric vector
%       Second time series.
%   Band : int, optional (default -1)
%       Sakoe-Chiba band width. Use -1 for full (unconstrained) DTW.
%   Metric : char, optional (default 'l1')
%       Distance metric for pointwise comparison. Currently supports 'l1'.
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
%   See also dtwc.compute_distance_matrix, dtwc.DTWClustering

    p = inputParser;
    addRequired(p, 'x', @isnumeric);
    addRequired(p, 'y', @isnumeric);
    addParameter(p, 'Band', -1, @(v) isnumeric(v) && isscalar(v));
    addParameter(p, 'Metric', 'l1', @ischar);
    parse(p, x, y, varargin{:});

    d = dtwc_mex('dtw_distance', ...
                  double(x(:)'), ...
                  double(y(:)'), ...
                  int32(p.Results.Band), ...
                  p.Results.Metric);
end
