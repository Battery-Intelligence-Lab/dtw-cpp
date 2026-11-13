function s = settings(varargin)
%SETTINGS Create or refine a CasADi-style DTWC++ workflow settings struct.
%
%   s = dtwc.settings('K', 3, 'Method', 'pam', 'Device', 'cpu')
%   s = dtwc.settings('K', 3, 'Output', '')   % pure in-memory (no files written)
%   s2 = dtwc.settings(s, 'Device', 'gpu')    % override an existing settings struct
%
%   Workflow:
%     s   = dtwc.settings('K', 3, 'Method', 'pam', 'Output', './results');
%     ds  = dtwc.load_data('Crop_TRAIN.tsv', s);
%     res = dtwc.do_clustering(ds, s);
%     dtwc.write_data(res, './archive');      % optional extra export
%
%   See also dtwc.load_data, dtwc.do_clustering, dtwc.write_data

%> @file settings.m
%> @brief Unified CasADi-style settings constructor.
%> @author Volkan Kumtepeli

    if ~isempty(varargin) && isstruct(varargin{1}) && isfield(varargin{1}, 'IsDtwcSettings')
        s = varargin{1};
        args = varargin(2:end);
    else
        s = struct( ...
            'IsDtwcSettings', true, ...
            'Input', '', ...
            'SkipCols', 0, ...
            'SkipRows', 0, ...
            'Delimiter', '', ...
            'Name', '', ...
            'Dtype', 'float64', ...
            'Ndim', 1, ...
            'Output', './results', ...
            'Checkpoint', '', ...
            'CheckpointInterval', 0, ...
            'K', 0, ...
            'ClusterArgs', {{}} ...
        );
        args = varargin;
    end

    [args{:}] = convertStringsToChars(args{:});
    if mod(numel(args), 2) ~= 0
        error('dtwc:invalidArgument', 'settings: options must be given as Name-Value pairs.');
    end

    clusterMap = containers.Map('KeyType', 'char', 'ValueType', 'any');
    order = {};
    for i = 1:2:numel(s.ClusterArgs)
        lk = lower(s.ClusterArgs{i});
        clusterMap(lk) = {s.ClusterArgs{i}, s.ClusterArgs{i + 1}};
        order{end + 1} = lk; %#ok<AGROW>
    end

    for i = 1:2:numel(args)
        key = args{i};
        val = args{i + 1};
        if ~(ischar(key) && (isempty(key) || isrow(key)))
            error('dtwc:invalidArgument', 'settings: option name must be a string.');
        end
        switch lower(key)
            case 'input'
                s.Input = char(val);
            case 'skipcols'
                s.SkipCols = val;
            case 'skiprows'
                s.SkipRows = val;
            case 'delimiter'
                s.Delimiter = char(val);
            case 'dtype'
                s.Dtype = dtwc_mex('dtype', char(val));
            case 'ndim'
                if ~(isnumeric(val) && isscalar(val) && isreal(val) && isfinite(val) && val >= 1 && val == fix(val))
                    error('dtwc:invalidArgument', 'settings: Ndim must be a positive integer.');
                end
                s.Ndim = double(val);
            case 'output'
                if isempty(val)
                    s.Output = '';
                else
                    s.Output = char(val);
                end
            case 'checkpoint'
                if isempty(val)
                    s.Checkpoint = '';
                else
                    s.Checkpoint = char(val);
                end
            case 'checkpointinterval'
                if ~(isnumeric(val) && isscalar(val) && isreal(val) && isfinite(val) && val >= 0 && val == fix(val))
                    error('dtwc:invalidArgument', 'settings: CheckpointInterval must be a non-negative integer.');
                end
                s.CheckpointInterval = double(val);
            case {'k', 'nclusters', 'n_clusters'}
                if ~(isnumeric(val) && isscalar(val) && isreal(val) && isfinite(val) && val >= 1 && val == fix(val))
                    error('dtwc:invalidArgument', 'settings: K must be a positive integer.');
                end
                s.K = double(val);
            case 'name'
                s.Name = char(val);
                lk = 'name';
                if ~isKey(clusterMap, lk)
                    order{end + 1} = lk; %#ok<AGROW>
                end
                clusterMap(lk) = {'Name', s.Name};
            otherwise
                lk = lower(key);
                if ~isKey(clusterMap, lk)
                    order{end + 1} = lk; %#ok<AGROW>
                end
                clusterMap(lk) = {key, val};
        end
    end

    rebuilt = cell(1, 2 * numel(order));
    for idx = 1:numel(order)
        pair = clusterMap(order{idx});
        rebuilt{2 * idx - 1} = pair{1};
        rebuilt{2 * idx} = pair{2};
    end
    s.ClusterArgs = rebuilt;

    % Validate clustering keys immediately in C++ (using k=1 placeholder if K not yet set)
    checkK = s.K;
    if checkK < 1
        checkK = 1;
    end
    prob = dtwc.Problem();
    cleaner = onCleanup(@() delete(prob));
    dtwc_mex('apply', prob.get_handle(), checkK, s.ClusterArgs{:});
end

