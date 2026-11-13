function D = cdist(X, varargin)
%CDIST Cross-collection (n_X x n_Y) DTW distance matrix in C++/SIMD/CUDA.
%
%   D = dtwc.distance.cdist(X, Y)
%   D = dtwc.distance.cdist(X)                          % pairwise X vs X
%   D = dtwc.distance.cdist(X, Y, 'Band', 10, 'Metric', 'squared_euclidean')
%   D = dtwc.distance.cdist(X, Y, 'Device', 'gpu', 'GpuPrecision', 'fp32')
%
%   X and Y may be an N x L numeric matrix, a cell of numeric vectors, a
%   dtwc.Dataset, or a file/folder path.
%
%   See also dtwc.distance.dtw, dtwc.cdist, dtwc.cluster

%> @file cdist.m
%> @brief Cross-collection DTW distance matrix.
%> @author Volkan Kumtepeli

    if isempty(varargin)
        Y = [];
        opts = {};
    elseif (ischar(varargin{1}) || (isstring(varargin{1}) && isscalar(varargin{1}))) ...
            && mod(numel(varargin), 2) == 0 && ~isfile(varargin{1}) && ~isfolder(varargin{1})
        Y = [];
        opts = varargin;
    else
        Y = varargin{1};
        opts = varargin(2:end);
    end

    [opts{:}] = convertStringsToChars(opts{:});
    if mod(numel(opts), 2) ~= 0
        error('dtwc:invalidArgument', 'cdist settings must come in Name-Value pairs.');
    end

    ndim = 1;
    dtype = 'float64';
    keep = true(1, numel(opts));
    for i = 1:2:numel(opts)
        if strcmpi(opts{i}, 'Ndim')
            ndim = opts{i + 1};
            keep(i:i+1) = false;
        elseif strcmpi(opts{i}, 'Dtype')
            dtype = dtwc_mex('dtype', char(opts{i + 1}));
            keep(i:i+1) = false;
        end
    end
    mexOpts = opts(keep);

    [xSeries, xDtype, xNdim] = materialize_for_cdist(X, dtype, ndim);
    if isempty(Y)
        ySeries = xSeries;
        yDtype = xDtype;
        yNdim = xNdim;
    else
        [ySeries, yDtype, yNdim] = materialize_for_cdist(Y, dtype, ndim);
    end
    if xNdim ~= yNdim
        error('dtwc:invalidArgument', 'cdist: X.Ndim (%d) and Y.Ndim (%d) must match.', xNdim, yNdim);
    end

    D = dtwc_mex('cdist', cast_for_cdist(xSeries, xDtype), cast_for_cdist(ySeries, yDtype), ...
                 double(xNdim), xDtype, yDtype, mexOpts{:});
end

function series = cast_for_cdist(series, dtype)
    type = 'double';
    if strcmp(dtype, 'float32')
        type = 'single';
    end
    if iscell(series)
        numeric = cellfun(@isnumeric, series);
        series(numeric) = cellfun(@(s) cast(s, type), series(numeric), 'UniformOutput', false);
    elseif isnumeric(series)
        series = cast(series, type);
    end
end

function [series, dtype, effNdim] = materialize_for_cdist(src, defaultDtype, requestedNdim)
    if isa(src, 'dtwc.Dataset')
        [series, ~, dsNdim] = src.as_series();
        dtype = src.Dtype;
        effNdim = dsNdim;
        if requestedNdim ~= 1 && dsNdim ~= requestedNdim
            error('dtwc:invalidArgument', 'cdist: Dataset Ndim (%d) disagrees with Ndim (%d).', ...
                  dsNdim, requestedNdim);
        end
    elseif ischar(src) || (isstring(src) && isscalar(src))
        ds = dtwc.Dataset(char(src), 0, 0, '', '', defaultDtype, requestedNdim);
        [series, ~, effNdim] = ds.as_series();
        dtype = ds.Dtype;
    else
        series = src;
        dtype = defaultDtype;
        effNdim = requestedNdim;
    end
end

