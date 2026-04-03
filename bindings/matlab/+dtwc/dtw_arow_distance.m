function d = dtw_arow_distance(x, y, varargin)
%DTW_AROW_DISTANCE Compute DTW-AROW distance (diagonal-only for missing).
%
%   d = dtwc.dtw_arow_distance(x, y)
%   d = dtwc.dtw_arow_distance(x, y, 'Band', 10)
%
%   DTW-AROW restricts the warping path to diagonal-only alignment when
%   either series has a NaN value, preventing many-to-one "free stretching"
%   through missing regions. More principled than zero-cost missing.
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
%       The DTW-AROW distance.
%
%   Reference: Yurtman et al. (2023), "Estimating DTW Distance Between
%              Time Series with Missing Data" (ECML-PKDD 2023)
%
%   See also dtwc.dtw_distance_missing, dtwc.dtw_distance

    p = inputParser;
    addRequired(p, 'x', @isnumeric);
    addRequired(p, 'y', @isnumeric);
    addParameter(p, 'Band', -1, @(v) isnumeric(v) && isscalar(v));
    parse(p, x, y, varargin{:});

    d = dtwc_mex('dtw_arow_distance', ...
                  double(x(:)'), ...
                  double(y(:)'), ...
                  double(p.Results.Band));
end
