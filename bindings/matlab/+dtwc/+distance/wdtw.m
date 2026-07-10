function d = wdtw(x, y, varargin)
%WDTW Weighted DTW distance.
    p = inputParser;
    addRequired(p, 'x', @isnumeric);
    addRequired(p, 'y', @isnumeric);
    addParameter(p, 'Band', -1, @(v) isnumeric(v) && isscalar(v));
    addParameter(p, 'G', 0.05, @(v) isnumeric(v) && isscalar(v));
    parse(p, x, y, varargin{:});

    if ~isfinite(p.Results.G) || p.Results.G < 0
        error('dtwc:invalidArgument', ...
              'WDTW g must be finite and non-negative.');
    end

    d = dtwc_mex('wdtw_distance', ...
                 double(x(:)'), ...
                 double(y(:)'), ...
                 double(p.Results.Band), ...
                 double(p.Results.G));
end
