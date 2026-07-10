function metric = validate_metric(value, entryPoint)
%VALIDATE_METRIC Normalize the MATLAB distance metric vocabulary.
%   MATLAB's current direct kernels implement L1 only. Known squared aliases
%   remain a typed capability error; every unknown token is an invalid argument
%   and must never fall through to L1.
    if isstring(value)
        if ~isscalar(value)
            error('dtwc:invalidArgument', 'Metric must be a scalar string.');
        end
        value = char(value);
    end
    if ~ischar(value) || size(value, 1) ~= 1
        error('dtwc:invalidArgument', 'Metric must be a character row vector.');
    end

    metric = lower(value);
    if strcmp(metric, 'l1')
        return;
    end
    if any(strcmp(metric, {'squared_euclidean', 'sqeuclidean'}))
        error('dtwc:distance:unsupportedMetric', ...
              'MATLAB distance.%s currently supports Metric=''l1'' only.', ...
              entryPoint);
    end
    error('dtwc:invalidArgument', ...
          'Unknown metric ''%s''. Expected one of: l1, squared_euclidean, sqeuclidean.', ...
          value);
end
