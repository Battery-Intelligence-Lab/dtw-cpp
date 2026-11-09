function res = cluster(data, k, varargin)
%CLUSTER Cluster time series into k groups with DTW; returns a dtwc.Result.
%
%   res = dtwc.cluster(data, k)
%   res = dtwc.cluster(data, k, 'Method', 'clara', 'Band', 10, 'MaxIter', 50)
%
%   data : a dtwc.Dataset (dtwc.load), a file or folder path, an N x L numeric
%          matrix (one series per row) or a cell of numeric vectors (one series
%          each, any lengths).
%   k    : the number of clusters.
%
%   The run is C++ dtwc::run, the run dtwc_cl makes, without its files: C++
%   reads a path itself (CSV/TSV text, a folder of it or an Arrow IPC file,
%   which its memory plan may stream through CLARA), MATLAB's parquetread reads
%   Parquet (dtwc.load), and series in memory go as they are.
%
%   Dtype is dtwc.load's, for a path or series in memory that cluster loads:
%   'float32' stores the series as single, half the memory.
%
%   The name-value pairs are these dtwc_cl keys, by their long names in
%   CamelCase, the words of Python's cluster() keywords:
%   Name, MatrixDtype, RamLimit, Method, Band, Metric, Variant, MaxIter, NInit,
%   Dc, WdtwG, AdtwPenalty, SdtwGamma, MsmC, TweNu, TweLambda, MvMode,
%   MissingStrategy, SampleSize, NSamples, Seed, BatchSize, Linkage, Solver,
%   MipGap, TimeLimit, NoWarmStart, NumericFocus, MipFocus, VerboseSolver,
%   LrMaxNodes, LrRoot, Device, GpuPrecision, Verbose, NoReport ('float32'
%   MatrixDtype stores the distance matrix at half the memory, each distance
%   rounded to a single; Verbose prints dtwc_cl -v's lines, the memory plan
%   among them; NoReport true makes no run report, so Result.Report is '').
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
%
%   See also dtwc.load, dtwc.Result, dtwc.device

%> @file cluster.m
%> @brief High-level clustering entry point.
%> @author Volkan Kumtepeli

    [varargin{:}] = convertStringsToChars(varargin{:});
    key = 2 * find(strcmpi(varargin(1:2:end - 1), 'Dtype')) - 1;   % load's, not the run's
    loadArgs = varargin(sort([key, key + 1]));
    varargin([key, key + 1]) = [];
    data = dtwc.load(data, loadArgs{:});
    if ischar(data.Source) && isempty(dtwc_mex('parquet_files', data.Source))
        % C++ reads the path, after it has checked the keys.
        out = dtwc_mex('run', data.Source, {}, data.SkipCols, data.SkipRows, data.Delimiter, data.Dtype, ...
                       data.Name, k, varargin{:});
    else
        % The keys are checked before parquetread reads a file, as for a path.
        settings = dtwc.Problem();
        dtwc_mex('apply', settings.get_handle(), k, varargin{:});
        delete(settings);
        [series, names] = data.as_series();
        out = dtwc_mex('run', cast_series(series, data.Dtype), names, 0, 0, '', data.Dtype, data.Name, k, ...
                       varargin{:});
    end
    res = dtwc.Result(out);
end
