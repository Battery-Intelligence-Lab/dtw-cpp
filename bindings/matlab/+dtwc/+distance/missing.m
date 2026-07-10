function d = missing(x, y, varargin)
%MISSING Zero-cost DTW distance for NaN-containing inputs.
    p = inputParser;
    addRequired(p, 'x', @isnumeric);
    addRequired(p, 'y', @isnumeric);
    addParameter(p, 'Band', -1, @(v) isnumeric(v) && isscalar(v));
    addParameter(p, 'Metric', 'l1', @(v) ischar(v) || isstring(v));
    parse(p, x, y, varargin{:});

    validate_metric(p.Results.Metric, 'missing');

    d = dtwc_mex('dtw_distance_missing', ...
                 double(x(:)'), ...
                 double(y(:)'), ...
                 double(p.Results.Band));
end
