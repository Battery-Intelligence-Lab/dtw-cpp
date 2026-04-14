function d = dtw(x, y, varargin)
%DTW Convenience dispatcher for DTW-family distances.
%
%   d = dtwc.distance.dtw(x, y)
%   d = dtwc.distance.dtw(x, y, 'Variant', 'wdtw', 'G', 0.1)
%   d = dtwc.distance.dtw(x, y, 'MissingStrategy', 'arow')
%
%   This dispatcher is intended for interactive use and examples. For tight
%   loops, prefer the explicit functions under dtwc.distance.*.

    p = inputParser;
    addParameter(p, 'Variant', 'standard', @ischar);
    addParameter(p, 'Band', -1, @(v) isnumeric(v) && isscalar(v));
    addParameter(p, 'Metric', 'l1', @ischar);
    addParameter(p, 'G', 0.05, @(v) isnumeric(v) && isscalar(v));
    addParameter(p, 'Penalty', 1.0, @(v) isnumeric(v) && isscalar(v));
    addParameter(p, 'Gamma', 1.0, @(v) isnumeric(v) && isscalar(v));
    addParameter(p, 'MissingStrategy', 'error', @ischar);
    parse(p, varargin{:});

    variant = lower(strrep(p.Results.Variant, '-', '_'));
    missing = lower(strrep(p.Results.MissingStrategy, '-', '_'));

    if ~strcmp(missing, 'error')
        if ~strcmp(variant, 'standard')
            error(['dtwc:distance:invalidCombination', ...
                   ' MissingStrategy dispatch currently requires Variant=''standard''. ', ...
                   'Use dtwc.Problem for combined variant + missing-data workflows.']);
        end
        switch missing
            case {'zero_cost', 'missing'}
                d = dtwc.distance.missing(x, y, 'Band', p.Results.Band, 'Metric', p.Results.Metric);
                return;
            case 'arow'
                d = dtwc.distance.arow(x, y, 'Band', p.Results.Band, 'Metric', p.Results.Metric);
                return;
            case 'interpolate'
                error(['dtwc:distance:notImplemented', ...
                       ' MissingStrategy=''interpolate'' is available through dtwc.Problem, ', ...
                       'not the standalone distance dispatcher.']);
            otherwise
                error('dtwc:distance:badMissingStrategy', ...
                      'Unknown MissingStrategy ''%s''.', p.Results.MissingStrategy);
        end
    end

    switch variant
        case {'standard', 'dtw'}
            d = dtwc.distance.standard(x, y, 'Band', p.Results.Band, 'Metric', p.Results.Metric);
        case 'ddtw'
            d = dtwc.distance.ddtw(x, y, 'Band', p.Results.Band);
        case 'wdtw'
            d = dtwc.distance.wdtw(x, y, 'Band', p.Results.Band, 'G', p.Results.G);
        case 'adtw'
            d = dtwc.distance.adtw(x, y, 'Band', p.Results.Band, 'Penalty', p.Results.Penalty);
        case {'softdtw', 'soft_dtw'}
            d = dtwc.distance.soft_dtw(x, y, 'Gamma', p.Results.Gamma);
        otherwise
            error('dtwc:distance:badVariant', ...
                  'Unknown Variant ''%s''.', p.Results.Variant);
    end
end
