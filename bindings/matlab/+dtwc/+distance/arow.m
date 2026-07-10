function d = arow(x, y, varargin)
%AROW DTW-AROW distance.
    p = inputParser;
    addRequired(p, 'x', @isnumeric);
    addRequired(p, 'y', @isnumeric);
    addParameter(p, 'Band', -1, @(v) isnumeric(v) && isscalar(v));
    addParameter(p, 'Metric', 'l1', @(v) ischar(v) || isstring(v));
    parse(p, x, y, varargin{:});

    validate_metric(p.Results.Metric, 'arow');

    d = dtwc_mex('dtw_arow_distance', ...
                 double(x(:)'), ...
                 double(y(:)'), ...
                 double(p.Results.Band));
end
