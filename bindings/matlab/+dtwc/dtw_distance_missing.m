function d = dtw_distance_missing(x, y, varargin)
%DTW_DISTANCE_MISSING Compute DTW distance with missing data support.
%
%   d = dtwc.dtw_distance_missing(x, y)
%   d = dtwc.dtw_distance_missing(x, y, 'Band', 10)
%
%   NaN values in x or y are treated as missing. Pairs where either value
%   is NaN contribute zero cost (the warping path passes through missing
%   regions without penalty).
%
%   Parameters
%   ----------
%   x : numeric vector (may contain NaN)
%       First time series.
%   y : numeric vector (may contain NaN)
%       Second time series.
%   Band : int, optional (default -1)
%       Sakoe-Chiba band width. Use -1 for full DTW.
%
%   Returns
%   -------
%   d : double
%       The DTW distance with missing data handling.
%
%   Reference: Yurtman et al. (2023), "Estimating DTW Distance Between
%              Time Series with Missing Data"
%
%   See also dtwc.dtw_arow_distance, dtwc.dtw_distance

    p = inputParser;
    addRequired(p, 'x', @isnumeric);
    addRequired(p, 'y', @isnumeric);
    addParameter(p, 'Band', -1, @(v) isnumeric(v) && isscalar(v));
    parse(p, x, y, varargin{:});

    d = dtwc_mex('dtw_distance_missing', ...
                  double(x(:)'), ...
                  double(y(:)'), ...
                  double(p.Results.Band));
end
