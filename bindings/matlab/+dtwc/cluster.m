%> @file cluster.m
%> @brief High-level clustering entry point (api-contract-2.0.md §1.3).
%> @author Volkan Kumtepeli
function res = cluster(data, k, varargin)
%CLUSTER Cluster time series into k groups; returns a dtwc.Result (contract §1.3).
%
%   res = dtwc.cluster(data, k)
%   res = dtwc.cluster(data, k, 'method','pam', 'band',-1, 'device','', 'max_iter',100)
%
%   Parameters
%   ----------
%   data : dtwc.Dataset, N x L numeric matrix, cell array of numeric vectors
%          (ragged in-memory source, one series per cell), or file path.
%   k : number of clusters (positive integer, at most the number of series).
%   method : 'auto' | 'pam' | 'onebatch' | 'clara' | 'kmedoids' | 'mip' |
%            'lrcore' | 'tadpole' | 'hierarchical' (alias 'hclust').
%            Default 'pam'. Unknown method -> 'dtwc:invalidArgument'.
%   band : Sakoe-Chiba band. -1 = full DTW. Default -1.
%   device : per-call device override ('' = keep the process device). The
%            override is local to this call and never mutates dtwc.device().
%   max_iter : maximum iterations. Default 100.
%
%   This function parses arguments and makes ONE gateway call. Method routing,
%   the k <= N guard, the device override and the source semantics
%   (skip_cols/skip_rows/delimiter/name) are all decided by C++ dtwc::cluster(),
%   so MATLAB cannot drift from the reference implementation.
%
%   See also dtwc.load, dtwc.Result, dtwc.device

    p = inputParser;
    addRequired(p, 'data');
    addRequired(p, 'k', @(v) isnumeric(v) && isscalar(v) && v > 0);
    addParameter(p, 'method', 'pam', @(v) ischar(v) || isstring(v));
    addParameter(p, 'band', -1, @(v) isnumeric(v) && isscalar(v));
    addParameter(p, 'device', '', @(v) ischar(v) || isstring(v));
    addParameter(p, 'max_iter', 100, @(v) isnumeric(v) && isscalar(v) && v > 0);
    parse(p, data, k, varargin{:});

    if isa(data, 'dtwc.Dataset')
        source    = data.Source;
        skip_cols = data.SkipCols;
        skip_rows = data.SkipRows;
        delimiter = data.Delimiter;
        name      = data.Name;
    elseif isnumeric(data) || iscell(data) || ischar(data) || isstring(data)
        source    = data;
        skip_cols = 0;
        skip_rows = 0;
        delimiter = '';
        name      = '';   % '' lets C++ derive the name (file stem / 'dataset')
    else
        error('dtwc:invalidArgument', ...
              ['cluster: data must be a dtwc.Dataset, a numeric matrix, a cell ' ...
               'array of numeric vectors, or a file path.']);
    end

    if ischar(source) || isstring(source)
        source = char(source);
    elseif iscell(source)
        % Ragged in-memory source: one cell per series, matching the C++
        % load(series_type) overload the Python list route already uses.
        source = dtwc.Dataset.normalise_cell_series(source, 'cluster');
    else
        source = double(source);
    end

    out = dtwc_mex('tier1_cluster', source, double(k), ...
                   char(p.Results.method), double(p.Results.band), ...
                   char(p.Results.device), double(p.Results.max_iter), ...
                   double(skip_cols), double(skip_rows), ...
                   char(delimiter), char(name));

    res = dtwc.Result(out);
end
