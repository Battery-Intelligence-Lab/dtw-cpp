%> @file DTWClustering.m
%> @brief K-medoids clustering with DTW distance via DTWC++.
%> @author Volkan Kumtepeli
classdef DTWClustering
%DTWCLUSTERING K-medoids clustering with DTW distance via DTWC++.
%
%   Mirrors the Python dtwcpp.DTWClustering API. Uses dtwc.Problem
%   internally for handle-based C++ object management.
%
%   obj = dtwc.DTWClustering('NClusters', 3, 'Band', 10)
%   obj = obj.fit(X);
%   labels = obj.Labels;
%
%   % Or use fit_predict for convenience:
%   labels = dtwc.DTWClustering('NClusters', 3).fit_predict(X);
%
%   Properties (configurable)
%   -------------------------
%   NClusters : int (default 3)
%       Number of clusters.
%   Band : int (default -1)
%       Sakoe-Chiba band width. -1 for full DTW.
%   Metric : char (default 'l1')
%       Pointwise cost metric: 'l1' or 'squared_euclidean'. A non-L1 metric
%       requires Variant 'standard' and MissingStrategy 'error'.
%   MaxIter : int (default 100)
%       Maximum iterations for the clustering algorithm.
%   NInit : int (default 1)
%       Number of random restarts.
%   Variant : char (default 'standard')
%       DTW variant: 'standard', 'ddtw', 'wdtw', 'adtw', 'softdtw'.
%   WdtwG : double (default 0.05)
%       Steepness parameter for WDTW.
%   AdtwPenalty : double (default 1.0)
%       Penalty for non-diagonal steps in ADTW.
%   MissingStrategy : char (default 'error')
%       Strategy for NaN values: 'error', 'zero_cost', 'arow', 'interpolate'.
%   Device : char (default '')
%       Per-call device override ('' = the process device). Validated through
%       dtwc.device and restored afterwards, so fit() never mutates it.
%
%   Properties (read-only, set after fit)
%   -------------------------------------
%   Labels : int32 row vector (1 x N)
%       Cluster assignments (1-based).
%   MedoidIndices : int32 row vector (1 x k)
%       Indices of medoid series (1-based).
%   TotalCost : double
%       Sum of intra-cluster DTW distances.
%
%   See also dtwc.Problem, dtwc.fast_pam, dtwc.distance.dtw
% @author Volkan Kumtepeli

    properties
        NClusters (1,1) {mustBePositive, mustBeInteger} = 3
        Band (1,1) {mustBeInteger} = -1
        Metric (1,:) char = 'l1'
        MaxIter (1,1) {mustBePositive, mustBeInteger} = 100
        NInit (1,1) {mustBePositive, mustBeInteger} = 1
        Variant (1,:) char = 'standard'
        WdtwG (1,1) double = 0.05
        AdtwPenalty (1,1) double = 1.0
        MissingStrategy (1,:) char = 'error'
        Device (1,:) char = ''
    end

    properties (SetAccess = private)
        Labels (:,:) int32 = int32([])
        MedoidIndices (:,:) int32 = int32([])
        TotalCost (1,1) double = NaN
    end

    methods
        function obj = DTWClustering(varargin)
        %DTWCLUSTERING Construct a DTWClustering object.
        %   obj = dtwc.DTWClustering()
        %   obj = dtwc.DTWClustering('NClusters', 5, 'Band', 10)
            p = inputParser;
            addParameter(p, 'NClusters', 3, @(v) isnumeric(v) && isscalar(v) && v > 0);
            addParameter(p, 'Band', -1, @(v) isnumeric(v) && isscalar(v));
            addParameter(p, 'Metric', 'l1', @ischar);
            addParameter(p, 'MaxIter', 100, @(v) isnumeric(v) && isscalar(v) && v > 0);
            addParameter(p, 'NInit', 1, @(v) isnumeric(v) && isscalar(v) && v > 0);
            addParameter(p, 'Variant', 'standard', @ischar);
            addParameter(p, 'WdtwG', 0.05, @(v) isnumeric(v) && isscalar(v));
            addParameter(p, 'AdtwPenalty', 1.0, @(v) isnumeric(v) && isscalar(v));
            addParameter(p, 'MissingStrategy', 'error', @ischar);
            addParameter(p, 'Device', '', @(v) ischar(v) || isstring(v));
            parse(p, varargin{:});

            obj.NClusters = p.Results.NClusters;
            obj.Band = p.Results.Band;
            obj.Metric = p.Results.Metric;
            obj.MaxIter = p.Results.MaxIter;
            obj.NInit = p.Results.NInit;
            obj.Variant = p.Results.Variant;
            obj.WdtwG = p.Results.WdtwG;
            obj.AdtwPenalty = p.Results.AdtwPenalty;
            obj.MissingStrategy = p.Results.MissingStrategy;
            obj.Device = char(p.Results.Device);
            obj.validate_variant_parameters();
        end

        function obj = fit(obj, X)
        %FIT Run k-medoids clustering on the data matrix X.
        %   obj = obj.fit(X)
        %
        %   Parameters
        %   ----------
        %   X : double matrix (N x L)
        %       Each row is a time series of length L.
            % Validate the executable distance contract before input/device or
            % Problem effects, including values changed after construction.
            obj.validate_variant_parameters();
            metric = obj.resolve_metric();
            validateattributes(X, {'numeric'}, {'2d', 'nonempty'}, 'fit', 'X');

            % Per-call device override (contract §1.5): resolved through the one
            % dtwc::Env registry for validation and normalisation, then restored,
            % so fit() never leaves the process device changed. An unknown device
            % / gpu-without-backend raises dtwc:deviceError (no silent fallback).
            if isempty(obj.Device)
                activeDevice = dtwc.device();
            else
                previousDevice = dtwc.device();
                deviceCleanup = onCleanup(@() dtwc.device(previousDevice));
                activeDevice = dtwc.device(obj.Device);
            end
            if strcmp(activeDevice, 'hpc')
                error('dtwc:deviceError', ...
                    ['DTWClustering: device=''hpc'' offloads the whole job and ' ...
                     'has no MATLAB transport. Use the Python API or ' ...
                     'scripts/slurm/slurm_remote.sh.']);
            end

            % Problem's lazy matrix is intrinsically L1, so a non-L1 metric needs
            % the exact matrix built up front -- the same rule the Python
            % estimator follows (dtwcpp/_clustering.py).
            precomputed = [];
            if ~strcmp(metric, 'l1')
                precomputed = dtwc_mex('DTWClustering_compute_distance_matrix', ...
                    double(X), double(obj.Band), metric);
            end

            bestCost = Inf;
            bestLabels = [];
            bestMedoids = [];
            baseSeed = dtwc.default_random_seed();
            if obj.NInit - 1 > flintmax - baseSeed
                error('dtwc:invalidArgument', ...
                    'NInit is too large to assign distinct exact MATLAB seeds.');
            end

            for rep = 1:obj.NInit
                % Create a Problem for each repetition
                prob = dtwc.Problem('DTWClustering');
                prob.set_data(double(X));
                prob.set_band(obj.Band);
                prob.set_max_iter(obj.MaxIter);
                prob.set_verbose(false);
                dtwc.DTWClustering.apply_device_strategy(prob, activeDevice);

                % Set DTW variant
                if ~strcmp(obj.Variant, 'standard')
                    switch obj.Variant
                        case 'wdtw'
                            prob.set_variant('wdtw', obj.WdtwG);
                        case 'adtw'
                            prob.set_variant('adtw', obj.AdtwPenalty);
                        otherwise
                            prob.set_variant(obj.Variant);
                    end
                end

                % Set missing strategy
                if ~strcmp(obj.MissingStrategy, 'error')
                    prob.set_missing_strategy(obj.MissingStrategy);
                end

                if ~isempty(precomputed)
                    prob.set_distance_matrix(precomputed);
                end

                % Run FastPAM
                result = dtwc.fast_pam(prob, obj.NClusters, ...
                    'MaxIter', obj.MaxIter, 'Seed', baseSeed + (rep - 1));

                if result.total_cost < bestCost
                    bestCost = result.total_cost;
                    bestLabels = result.labels;
                    bestMedoids = result.medoid_indices;
                end
            end

            obj.Labels = bestLabels;
            obj.MedoidIndices = bestMedoids;
            obj.TotalCost = bestCost;
        end

        function labels = fit_predict(obj, X)
        %FIT_PREDICT Fit and return cluster labels.
        %   labels = obj.fit_predict(X)
            obj = obj.fit(X);
            labels = obj.Labels;
        end

        function labels = predict(obj, X)
        %PREDICT Assign new data to nearest medoids (requires prior fit).
        %   labels = obj.predict(X)
        %
        %   Assigns each row of X to the cluster of the nearest medoid
        %   from the most recent fit() call.
            if isempty(obj.MedoidIndices)
                error('dtwc:notFitted', ...
                      'Model has not been fitted. Call fit() first.');
            end
            error('dtwc:notImplemented', ...
                  'predict() for new data is not yet implemented.');
        end
    end

    methods (Access = private)
        function metric = resolve_metric(obj)
        %RESOLVE_METRIC Normalise and validate Metric before any other effect.
        %   Returns the canonical token. The accepted set and the incompatible
        %   cross-products mirror the Python estimator (dtwcpp/_clustering.py).
            metric = lower(strtrim(char(obj.Metric)));
            if ~ismember(metric, {'l1', 'squared_euclidean'})
                error('dtwc:invalidArgument', ...
                      'Unknown Metric ''%s''. Expected one of: l1, squared_euclidean.', ...
                      char(obj.Metric));
            end
            if strcmp(metric, 'l1'), return; end
            if ~strcmp(obj.Variant, 'standard')
                error('dtwc:invalidArgument', ...
                      ['Metric ''%s'' is implemented only for Variant ''standard''; ' ...
                       'Variant ''%s'' has its intrinsic L1 cost.'], ...
                      metric, obj.Variant);
            end
            if ~strcmp(obj.MissingStrategy, 'error')
                error('dtwc:invalidArgument', ...
                      'Metric ''%s'' is not implemented with MissingStrategy ''%s''.', ...
                      metric, obj.MissingStrategy);
            end
        end

        function validate_variant_parameters(obj)
            if ~isfinite(obj.WdtwG) || obj.WdtwG < 0
                error('dtwc:invalidArgument', ...
                      'WDTW g must be finite and non-negative.');
            end
            if ~isfinite(obj.AdtwPenalty) || obj.AdtwPenalty < 0
                error('dtwc:invalidArgument', ...
                      'ADTW penalty must be finite and non-negative.');
            end
        end
    end

    methods (Static, Hidden)
        function index = gpu_index(deviceName)
        %GPU_INDEX Ordinal N of a canonical 'gpu:N'/'cuda:N' name (0 otherwise).
        %   Mirrors dtwc::Env::set_device, which parses the suffix into
        %   device_index() and reports it back through the canonical name.
            index = 0;
            name = char(deviceName);
            colon = strfind(name, ':');
            if isempty(colon), return; end
            index = str2double(name((colon(1) + 1):end));
            if ~isfinite(index) || index ~= fix(index) || index < 0
                error('dtwc:deviceError', ...
                    'DTWClustering: unknown device ''%s''.', name);
            end
        end

        function apply_device_strategy(prob, activeDevice)
        %APPLY_DEVICE_STRATEGY Make the Problem execute on the selected device.
        %   Mirrors C++ detail::configure_device (dtwc/api.cpp): the GPU ordinal
        %   of a 'gpu:N' selection reaches Problem::cuda_settings.device_id
        %   before the strategy is chosen, so 'gpu:1' does not run on GPU 0.
        %   Without this the Problem kept its CPU default and a 'gpu' request was
        %   silently honoured on the CPU (gap F40).
            if strcmp(activeDevice, 'cpu'), return; end
            info = dtwc_mex('system_check');
            if ~info.cuda && ~info.metal
                error('dtwc:deviceError', ...
                    ['DTWClustering: device ''%s'' was selected but this build ' ...
                     'has no GPU backend.'], activeDevice);
            end
            prob.set_cuda_settings(dtwc.DTWClustering.gpu_index(activeDevice));
            if info.cuda
                prob.set_distance_strategy('cuda');
            else
                prob.set_distance_strategy('metal');
            end
        end
    end
end
