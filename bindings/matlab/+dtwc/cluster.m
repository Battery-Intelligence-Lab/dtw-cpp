function res = cluster(data, k, varargin)
%CLUSTER Cluster time series into k groups with DTW; returns a dtwc.Result.
%
%   res = dtwc.cluster(data, k)
%   res = dtwc.cluster(data, k, 'Method', 'clara', 'Band', 10, 'MaxIter', 50)
%   res = dtwc.cluster(data, k, 'Output', 'results', 'Checkpoint', 'ckpt')
%
%   data : a dtwc.Dataset (dtwc.load), a file or folder path, an N x L numeric
%          matrix (one series per row) or a cell of numeric vectors (one series
%          each, any lengths).
%   k    : the number of clusters.
%
%   The run is C++ dtwc::run, the run dtwc_cl makes, without its files unless
%   Output asks for them: C++ reads a path itself (CSV/TSV text, a folder of it
%   or an Arrow IPC file, which its memory plan may stream through CLARA),
%   MATLAB's parquetread reads Parquet (dtwc.load), and series in memory go as
%   they are.
%
%   Dtype is dtwc.load's, for a path or series in memory that cluster loads:
%   'float32' stores the series as single, half the memory. A dtwc.Dataset keeps
%   its own: a Dtype given that differs is refused. A multivariate input is made
%   by dtwc.load, which takes Ndim: dtwc.cluster(dtwc.load(X, 'Ndim', 2), k).
%
%   The name-value pairs are these dtwc_cl keys, by their long names in
%   CamelCase, the words of Python's cluster() keywords:
%   Name, MatrixDtype, RamLimit, Method, Band, Metric, Variant, MaxIter, NInit,
%   Dc, WdtwG, AdtwPenalty, SdtwGamma, MsmC, TweNu, TweLambda, MvMode,
%   MissingStrategy, SampleSize, NSamples, Seed, BatchSize, Linkage, Solver,
%   MipGap, TimeLimit, NoWarmStart, NumericFocus, MipFocus, VerboseSolver,
%   LrMaxNodes, Device, GpuPrecision, GpuShare, Fallback, Verbose, NoReport
%   ('float32' MatrixDtype stores the distance matrix at half the memory, each
%   distance rounded to a single; GpuShare is the GPU's share of the work Device
%   'cpu+gpu' splits with the CPU, -1 (the default) measured, [0, 1] forced so a
%   run repeats; Fallback true runs a Device that cannot be honoured (no GPU
%   backend or device, a Variant, MissingStrategy, GpuPrecision or number of
%   features the kernels lack, a Method with no GPU path) on the CPU instead of
%   raising dtwc:deviceError, one line on stderr and Result.Report saying why and
%   Result.device 'cpu' (this call only: dtwc.device has none; a failure once
%   the GPU has begun is an error either way); Verbose prints dtwc_cl -v's lines,
%   the memory plan among them; NoReport true makes no run report, so
%   Result.Report is '').
%   RamLimit is the memory the run plans for, bytes or a size such as '8G'; 0,
%   the default, plans with 90 % of the memory available (a MEX built without
%   llfio cannot tell it, and plans nothing): a limit that cannot hold the run
%   raises dtwc:invalidArgument, and an Arrow IPC file whose series it cannot
%   hold streams through CLARA. C++ reads and checks the keys
%   (dtwc::Config, then dtwc::apply) before the series are read; a key not given
%   takes dtwc_cl's default, so Method is 'auto' (PAM on a GPU and for up to
%   5000 series on the CPU, CLARA above, and CLARA too where PAM's distance
%   matrix does not fit the memory plan). Device is dtwc.device() unless given,
%   and a Device given sets this run's device only. Keys and names match
%   ignoring case. An unknown key, or a name no C++ table holds, raises
%   dtwc:invalidArgument naming the valid ones (a device name: dtwc:deviceError).
%   A name given twice takes its last value, so options kept in a cell array are
%   overridden by a pair after them: dtwc.cluster(ds, 3, opts{:}, 'Band', 2).
%
%   Three more names are the run's files, not keys of its Config. Output is the
%   directory the run writes its result files into, the files dtwc_cl writes,
%   then the run report ('' (the default) writes none: Result.save writes them
%   on request). Checkpoint is the directory of the distance matrix's <name>.dtwm
%   file, which the run saves there and resumes from where it holds one for these
%   data; CheckpointInterval saves it every that many rows (0, the default: once,
%   at the end; it needs Checkpoint).
%
%   See also dtwc.load, dtwc.Result, dtwc.device

%> @file cluster.m
%> @brief High-level clustering entry point.
%> @author Volkan Kumtepeli

    [varargin{:}] = convertStringsToChars(varargin{:});
    key = 2 * find(strcmpi(varargin(1:2:end - 1), 'Dtype')) - 1;   % load's, not the run's
    loadArgs = varargin(sort([key, key + 1]));
    varargin([key, key + 1]) = [];
    [output, varargin] = take_last(varargin, 'Output', '');         % the run's files, not keys of its Config
    [checkpoint, varargin] = take_last(varargin, 'Checkpoint', '');
    [interval, varargin] = take_last(varargin, 'CheckpointInterval', 0);
    if ~(isnumeric(interval) && isscalar(interval) && isreal(interval) && isfinite(interval) ...
            && interval >= 0 && interval == fix(interval))
        error('dtwc:invalidArgument', 'cluster: CheckpointInterval must be a non-negative integer.');
    end
    data = dtwc.load(data, loadArgs{:});
    if ischar(data.Source) && data.Ndim == 1 && isempty(dtwc_mex('parquet_files', data.Source))
        % C++ reads the path, after it has checked the keys.
        out = dtwc_mex('run_workflow', data.Source, {}, 1, data.SkipCols, data.SkipRows, data.Delimiter, ...
                       data.Dtype, data.Name, output, checkpoint, interval, k, varargin{:});
    else
        % The keys are checked before parquetread reads a file, as for a path. The series go with their Ndim.
        settings = dtwc.Problem();
        dtwc_mex('apply', settings.get_handle(), k, varargin{:});
        delete(settings);
        [series, names, ndim] = data.as_series();
        out = dtwc_mex('run_workflow', cast_series(series, data.Dtype), names, ndim, 0, 0, '', data.Dtype, ...
                       data.Name, output, checkpoint, interval, k, varargin{:});
    end
    res = dtwc.Result(out);
end

function [value, args] = take_last(args, name, value)
%TAKE_LAST The last value given for NAME among the name-value pairs ARGS (else VALUE), and ARGS without them.
    named = find(strcmpi(args(1:2:end - 1), name));
    if ~isempty(named)
        value = args{2 * named(end)};
    end
    args([2 * named - 1, 2 * named]) = [];
end
