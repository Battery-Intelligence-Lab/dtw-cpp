function d = soft_dtw(x, y, varargin)
%SOFT_DTW Soft-DTW distance.
    p = inputParser;
    addRequired(p, 'x', @isnumeric);
    addRequired(p, 'y', @isnumeric);
    addParameter(p, 'Gamma', 1.0, @(v) isnumeric(v) && isscalar(v));
    parse(p, x, y, varargin{:});

    if ~isfinite(p.Results.Gamma) || p.Results.Gamma <= 0
        error('dtwc:invalidArgument', ...
              'Soft-DTW gamma must be finite and positive.');
    end

    d = dtwc_mex('soft_dtw_distance', ...
                 double(x(:)'), ...
                 double(y(:)'), ...
                 double(p.Results.Gamma));
end
