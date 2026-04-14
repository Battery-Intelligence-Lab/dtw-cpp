function d = arow(x, y, varargin)
%AROW DTW-AROW distance.
    p = inputParser;
    addRequired(p, 'x', @isnumeric);
    addRequired(p, 'y', @isnumeric);
    addParameter(p, 'Band', -1, @(v) isnumeric(v) && isscalar(v));
    addParameter(p, 'Metric', 'l1', @(v) ischar(v) || isstring(v));
    parse(p, x, y, varargin{:});

    metric = lower(char(p.Results.Metric));
    if ~strcmp(metric, 'l1')
        error('dtwc:distance:unsupportedMetric', ...
              'MATLAB distance.arow currently supports Metric=''l1'' only.');
    end

    d = dtwc_mex('dtw_arow_distance', ...
                 double(x(:)'), ...
                 double(y(:)'), ...
                 double(p.Results.Band));
end
