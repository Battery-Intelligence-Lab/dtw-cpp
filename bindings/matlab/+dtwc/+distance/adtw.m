function d = adtw(x, y, varargin)
%ADTW Amerced DTW distance.
    p = inputParser;
    addRequired(p, 'x', @isnumeric);
    addRequired(p, 'y', @isnumeric);
    addParameter(p, 'Band', -1, @(v) isnumeric(v) && isscalar(v));
    addParameter(p, 'Penalty', 1.0, @(v) isnumeric(v) && isscalar(v));
    parse(p, x, y, varargin{:});

    d = dtwc_mex('adtw_distance', ...
                 double(x(:)'), ...
                 double(y(:)'), ...
                 double(p.Results.Band), ...
                 double(p.Results.Penalty));
end
