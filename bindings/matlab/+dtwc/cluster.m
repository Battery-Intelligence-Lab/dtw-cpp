%> @file cluster.m
%> @brief High-level clustering entry point (api-contract-2.0.md §1.3).
%> @author Volkan Kumtepeli
function res = cluster(data, k, varargin)
%CLUSTER Cluster time series into k groups with DTW; returns a dtwc.Result (contract §1.3).
%
%   res = dtwc.cluster(data, k)
%   res = dtwc.cluster(data, k, 'Method', 'clara', 'Band', 10, 'MaxIter', 50)
%
%   data : a dtwc.Dataset (dtwc.load), a file or folder path, an N x L numeric
%          matrix (one series per row) or a cell of numeric vectors (one series
%          each, any lengths).
%   k    : the number of clusters.
%
%   The name-value pairs are the dtwc_cl keys that are not about files, by
%   their long names in CamelCase, the words of Python's cluster() keywords:
%   Name, Method, Band, Metric, Variant, MaxIter, NInit, Dc, WdtwG,
%   AdtwPenalty, SdtwGamma, MsmC, TweNu, TweLambda, MvMode, MissingStrategy,
%   SampleSize, NSamples, Seed, BatchSize, Linkage, Solver, MipGap, TimeLimit,
%   NoWarmStart, NumericFocus, MipFocus, VerboseSolver, LrMaxNodes, Device,
%   GpuPrecision, Verbose. C++ reads and checks them (dtwc::Config, then
%   dtwc::apply) before the series are read; a key not given takes dtwc_cl's
%   default, so Method is 'auto' (PAM on a GPU and for up to 5000 series on
%   the CPU, CLARA above). Device is dtwc.device() unless given, and a Device
%   given sets this run's device only. An unknown key or value raises
%   dtwc:invalidArgument naming the valid ones.
%
%   See also dtwc.load, dtwc.Result, dtwc.device

    [varargin{:}] = convertStringsToChars(varargin{:});
    data = dtwc.load(data);
    prob = dtwc.Problem(data.Name);
    device = dtwc_mex('apply', prob.get_handle(), k, varargin{:});
    [series, names] = data.as_series();
    prob.set_data(series, names);
    res = dtwc.Result(prob, prob.cluster(), device);
end
