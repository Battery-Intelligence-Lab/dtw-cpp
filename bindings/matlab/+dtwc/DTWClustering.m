%> @file DTWClustering.m
%> @brief K-medoids clustering with DTW distance via DTWC++.
%> @author Volkan Kumtepeli
classdef DTWClustering
%DTWCLUSTERING K-medoids clustering with DTW distance, Python's dtwcpp.DTWClustering.
%
%   c = dtwc.DTWClustering('NClusters', 3, 'Band', 10);
%   c = c.fit(X);               % X: N x L numeric matrix (a series per row)
%                               %    or a cell of numeric vectors (any lengths)
%   c.Labels, c.MedoidIndices, c.Inertia
%   labels = c.predict(Y);      % the nearest medoid of each series of Y
%
%   fit hands C++ one dtwc.Problem, set up from these properties as
%   dtwc.cluster sets one up from its keys, and Problem.cluster runs Method
%   and its NInit seeded restarts on one distance matrix. predict, transform
%   and score read the fitted medoids; nothing is refitted.
%
%   Properties: Python's parameters in CamelCase. One left empty takes the C++
%   default (dtwc_cl's), and C++ checks every value when fit runs, so a value
%   no run can take (MaxIter = 0, an unknown Metric) raises dtwc:invalidArgument.
%     NClusters (3), Method ('pam'): any dtwc.cluster method.
%     Variant, Band, Metric, MissingStrategy, WdtwG, AdtwPenalty, MsmC, TweNu,
%     TweLambda: the distance, as dtwc.distance.dtw takes it.
%     MaxIter, NInit, MvMode, BatchSize, Device: as dtwc.cluster's keys; Device
%     empty is dtwc.device().
%     RandomState: the seed of the first restart (restart i takes RandomState
%     + i - 1); empty is dtwc.default_random_seed().
%
%   Fitted (read-only): Labels (1-based cluster of each series), MedoidIndices
%   (1-based), Inertia (the total distance of the series to their medoids) and
%   ClusterCenters (the medoid series).
%
%   See also dtwc.cluster, dtwc.Problem, dtwc.distance.dtw

    properties
        NClusters = 3
        Method = 'pam'
        Variant = []
        Band = []
        MaxIter = []
        NInit = []
        WdtwG = []
        AdtwPenalty = []
        MsmC = []
        TweNu = []
        TweLambda = []
        MvMode = []
        MissingStrategy = []
        Metric = []
        BatchSize = []
        RandomState = []
        Device = []
    end

    properties (SetAccess = private)
        Labels = []
        MedoidIndices = []
        Inertia = NaN
        ClusterCenters = {}
    end

    properties (Access = private)
        FitDistance = {}   % the distance settings the medoids were fitted under
    end

    properties (Constant, Access = private)
        DistanceKeys = {'Variant', 'Band', 'Metric', 'MissingStrategy', 'WdtwG', 'AdtwPenalty', ...
                        'MsmC', 'TweNu', 'TweLambda'}
        RunKeys = {'Method', 'MaxIter', 'NInit', 'MvMode', 'BatchSize', 'Device'}
    end

    methods
        function obj = DTWClustering(varargin)
        %DTWCLUSTERING obj = dtwc.DTWClustering('NClusters', 5, 'Band', 10)
        %   Name-value pairs set the properties of those names (ASCII case ignored).
            if mod(numel(varargin), 2)
                error('dtwc:invalidArgument', 'DTWClustering: properties come in name-value pairs.');
            end
            mc = ?dtwc.DTWClustering;
            settable = {mc.PropertyList(strcmp({mc.PropertyList.SetAccess}, 'public')).Name};
            for i = 1:2:numel(varargin)
                match = strcmpi(settable, varargin{i});
                if ~any(match)
                    error('dtwc:invalidArgument', 'DTWClustering: unknown property ''%s''. Valid: %s.', ...
                          char(varargin{i}), strjoin(settable, ', '));
                end
                obj.(settable{match}) = varargin{i + 1};
            end
        end

        function obj = fit(obj, X)
        %FIT Cluster X: an N x L numeric matrix (a series per row) or a cell of
        %   numeric vectors. obj = obj.fit(X)
            prob = dtwc.Problem('dtw_clustering');
            settings = obj.given([obj.DistanceKeys, obj.RunKeys]);
            if ~isempty(obj.RandomState)
                settings = [settings, {'Seed', obj.RandomState}];
            end
            [settings{:}] = convertStringsToChars(settings{:});
            dtwc_mex('apply', prob.get_handle(), obj.NClusters, settings{:});
            prob.set_data(X);
            result = prob.cluster();
            obj.Labels = result.labels;
            obj.MedoidIndices = result.medoid_indices;
            obj.Inertia = result.total_cost;
            if iscell(X)
                obj.ClusterCenters = reshape(X(result.medoid_indices), 1, []);
            else
                obj.ClusterCenters = num2cell(X(result.medoid_indices, :), 2).';
            end
            obj.FitDistance = obj.given(obj.DistanceKeys);
        end

        function labels = fit_predict(obj, X)
        %FIT_PREDICT Fit and return the cluster labels. labels = obj.fit_predict(X)
            obj = obj.fit(X);
            labels = obj.Labels;
        end

        function D = transform(obj, X)
        %TRANSFORM The DTW distance of each series of X to each medoid (M x k),
        %   under the distance settings the medoids were fitted with.
            if isempty(obj.ClusterCenters)
                error('dtwc:notFitted', 'DTWClustering is not fitted; call fit first.');
            end
            if ~iscell(X)
                X = num2cell(X, 2);
            end
            D = zeros(numel(X), numel(obj.ClusterCenters));
            for i = 1:numel(X)
                for j = 1:numel(obj.ClusterCenters)
                    D(i, j) = dtwc.distance.dtw(X{i}, obj.ClusterCenters{j}, obj.FitDistance{:});
                end
            end
        end

        function labels = predict(obj, X)
        %PREDICT The cluster (1-based) of the nearest medoid of each series of X.
            [~, labels] = min(obj.transform(X), [], 2);
            labels = labels.';
        end

        function s = score(obj, X)
        %SCORE Minus the total distance of the series of X to their nearest
        %   medoids (larger is better).
            s = -sum(min(obj.transform(X), [], 2));
        end
    end

    methods (Access = private)
        function pairs = given(obj, names)
        %GIVEN The name-value pairs of the properties NAMES that are not empty.
            pairs = {};
            for i = 1:numel(names)
                if ~isempty(obj.(names{i}))
                    pairs(end + 1:end + 2) = {names{i}, obj.(names{i})};
                end
            end
        end
    end
end
