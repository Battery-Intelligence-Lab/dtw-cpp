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
%       Pointwise cost metric: 'l1' or 'squared_euclidean' (Variant 'standard'
%       or 'ddtw'). C++ reads and checks the distance settings, as
%       dtwc.distance.dtw does.
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
%   Labels : double row vector (1 x N)
%       Cluster assignments (1-based).
%   MedoidIndices : double row vector (1 x k)
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
        Labels (:,:) double = double([])
        MedoidIndices (:,:) double = double([])
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
        end

        function obj = fit(obj, X)
        %FIT Run k-medoids clustering on the data matrix X.
        %   obj = obj.fit(X)
        %
        %   Parameters
        %   ----------
        %   X : double matrix (N x L)
        %       Each row is a time series of length L.
            % C++ checks the distance settings before the data or the device is
            % touched: the distance of two one-sample series runs that check.
            settings = obj.distance_settings();
            dtwc.distance.dtw(0, 0, settings{:});
            validateattributes(X, {'numeric'}, {'2d', 'nonempty'}, 'fit', 'X');

            % Per-call device override (contract §1.5): resolved through C++
            % dtwc::device() for validation and normalisation, then restored,
            % so fit() never leaves the process device changed. An unknown device
            % / hpc / gpu-without-backend raises dtwc:deviceError (no silent fallback).
            if isempty(obj.Device)
                activeDevice = dtwc.device();
            else
                previousDevice = dtwc.device();
                deviceCleanup = onCleanup(@() dtwc.device(previousDevice));
                activeDevice = dtwc.device(obj.Device);
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
                prob.set_distance(settings{:});
                prob.set_max_iter(obj.MaxIter);
                prob.set_verbose(false);
                dtwc.DTWClustering.apply_device_strategy(prob, activeDevice);

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
        function settings = distance_settings(obj)
        %DISTANCE_SETTINGS The distance settings, as dtwc.distance.dtw and
        %   Problem.set_distance take them.
            settings = {'Variant', obj.Variant, 'Band', obj.Band, ...
                        'Metric', obj.Metric, 'MissingStrategy', obj.MissingStrategy, ...
                        'WdtwG', obj.WdtwG, 'AdtwPenalty', obj.AdtwPenalty};
        end
    end

    methods (Static, Hidden)
        function apply_device_strategy(prob, activeDevice)
        %APPLY_DEVICE_STRATEGY Make the Problem execute on the selected device.
        %   Problem::set_device (C++): the GPU ordinal of a 'gpu:N' selection
        %   reaches cuda_settings.device_id, so 'gpu:1' does not run on GPU 0,
        %   and a build without a GPU backend raises dtwc:deviceError. Without
        %   this the Problem kept its CPU default and a 'gpu' request was
        %   silently honoured on the CPU (gap F40).
            prob.set_device(activeDevice);
        end
    end
end
