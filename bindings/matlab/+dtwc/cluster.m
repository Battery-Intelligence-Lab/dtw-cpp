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
%   data : dtwc.Dataset, N x L numeric matrix, or file path.
%   k : number of clusters (positive integer).
%   method : 'pam' | 'kmedoids' | 'clara' | 'mip' | 'hierarchical' (alias 'hclust')
%            | 'auto'. Default 'pam'. Unknown method -> 'dtwc:invalidArgument'.
%   band : Sakoe-Chiba band. -1 = full DTW. Default -1.
%   device : per-call device override ('' = keep global). Delegates to dtwc::Env.
%   max_iter : maximum iterations. Default 100.
%
%   See also dtwc.load, dtwc.Result, dtwc.fast_pam

    p = inputParser;
    addRequired(p, 'data');
    addRequired(p, 'k', @(v) isnumeric(v) && isscalar(v) && v > 0);
    addParameter(p, 'method', 'pam', @(v) ischar(v) || isstring(v));
    addParameter(p, 'band', -1, @(v) isnumeric(v) && isscalar(v));
    addParameter(p, 'device', '', @(v) ischar(v) || isstring(v));
    addParameter(p, 'max_iter', 100, @(v) isnumeric(v) && isscalar(v) && v > 0);
    parse(p, data, k, varargin{:});

    method   = lower(strrep(char(p.Results.method), '-', '_'));
    dev      = char(p.Results.device);
    max_iter = double(p.Results.max_iter);

    % Per-call device override delegates to dtwc::Env (no silent fallback).
    if ~isempty(dev)
        dtwc.device(dev);
    end
    activeDevice = dtwc.device();   % normalised device name for Result.device

    % Materialise the data source into an N x L matrix (+ optional names).
    names = {};
    if isa(data, 'dtwc.Dataset')
        [X, names] = data.materialize();
        nm = data.Name;
    elseif isnumeric(data)
        X = double(data);
        nm = 'dataset';
    elseif ischar(data) || isstring(data)
        [X, names] = dtwc.load(data).materialize();
        [~, nm, ~] = fileparts(char(data));
    else
        error('dtwc:invalidArgument', ...
              'cluster: data must be a dtwc.Dataset, a numeric matrix, or a file path.');
    end

    prob = dtwc.Problem(nm);
    prob.set_band(double(p.Results.band));
    prob.set_max_iter(max_iter);
    if isempty(names)
        prob.set_data(X);
    else
        prob.set_data(X, names);
    end

    switch method
        case {'pam', 'kmedoids', 'auto'}
            r = dtwc.fast_pam(prob, k, 'MaxIter', max_iter, ...
                'Seed', dtwc.default_random_seed());
            labels = r.labels; medoids = r.medoid_indices; cost = r.total_cost;
        case 'clara'
            r = dtwc.fast_clara(prob, k, ...
                'Seed', dtwc.default_random_seed());
            labels = r.labels; medoids = r.medoid_indices; cost = r.total_cost;
        case 'mip'
            prob.set_method('mip');
            prob.cluster();
            labels = prob.labels(); medoids = prob.medoids(); cost = prob.find_total_cost();
        case {'hierarchical', 'hclust'}
            dend = dtwc.build_dendrogram(prob);
            r = dtwc.cut_dendrogram(dend, prob, k);
            labels = r.labels; medoids = r.medoid_indices; cost = r.total_cost;
        otherwise
            error('dtwc:invalidArgument', ...
                  ['Unknown method ''%s''. Valid: pam, kmedoids, clara, mip, ' ...
                   'hierarchical (hclust), auto.'], char(p.Results.method));
    end

    res = dtwc.Result(prob, labels, medoids, cost, activeDevice);
end
