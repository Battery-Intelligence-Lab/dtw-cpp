function d = dtw(x, y, varargin)
%DTW DTW-family distance of two series, computed by C++ (dtwc::distance::dtw).
%
%   d = dtwc.distance.dtw(x, y)
%   d = dtwc.distance.dtw(x, y, 'Band', 10, 'Metric', 'squared_euclidean')
%   d = dtwc.distance.dtw(x, y, 'Variant', 'wdtw', 'WdtwG', 0.1)
%   d = dtwc.distance.dtw(x, y, 'MissingStrategy', 'arow')
%
%   The settings are the dtwc_cl keys in CamelCase; one not given takes its
%   C++ default (Standard DTW, full band, L1, no missing values):
%     Variant          'standard', 'ddtw', 'wdtw', 'adtw', 'softdtw', 'msm', 'twe'
%     Band             -1 (full DTW) or a Sakoe-Chiba half-width >= 0
%     Metric           'l1' or 'squared_euclidean' (Standard DTW and DDTW)
%     MissingStrategy  'error', 'zero_cost', 'arow' or 'interpolate' (Standard
%                      DTW); NaN is a missing value under the last three
%     WdtwG, AdtwPenalty, SdtwGamma, MsmC, TweNu, TweLambda   the parameter of
%                      WDTW, ADTW, Soft-DTW, MSM and TWE
%   C++ reads and checks them: an unknown name, a parameter outside its domain,
%   a combination no kernel implements, or a value the strategy does not take
%   raises dtwc:invalidArgument.
    if ~isnumeric(x) || ~isnumeric(y)
        error('dtwc:invalidArgument', 'x and y must be numeric.');
    end
    [varargin{:}] = convertStringsToChars(varargin{:});
    d = dtwc_mex('dtw', double(x(:)'), double(y(:)'), varargin{:});
end
