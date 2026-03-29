classdef DTWClustering
%DTWCLUSTERING K-medoids clustering with DTW distance via DTWC++.
%
%   Mirrors the Python dtwcpp.DTWClustering API.
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
%       Pointwise distance metric.
%   MaxIter : int (default 100)
%       Maximum iterations for the clustering algorithm.
%   NInit : int (default 1)
%       Number of random restarts.
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
%   See also dtwc.dtw_distance, dtwc.compute_distance_matrix

    properties
        NClusters (1,1) {mustBePositive, mustBeInteger} = 3
        Band (1,1) {mustBeInteger} = -1
        Metric (1,:) char = 'l1'
        MaxIter (1,1) {mustBePositive, mustBeInteger} = 100
        NInit (1,1) {mustBePositive, mustBeInteger} = 1
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
            parse(p, varargin{:});

            obj.NClusters = p.Results.NClusters;
            obj.Band = p.Results.Band;
            obj.Metric = p.Results.Metric;
            obj.MaxIter = p.Results.MaxIter;
            obj.NInit = p.Results.NInit;
        end

        function obj = fit(obj, X)
        %FIT Run k-medoids clustering on the data matrix X.
        %   obj = obj.fit(X)
        %
        %   Parameters
        %   ----------
        %   X : double matrix (N x L)
        %       Each row is a time series of length L.
            validateattributes(X, {'numeric'}, {'2d', 'nonempty'}, 'fit', 'X');

            bestCost = Inf;
            bestLabels = [];
            bestMedoids = [];

            for rep = 1:obj.NInit
                [labels, medoids, cost] = dtwc_mex('cluster', ...
                    double(X), ...
                    int32(obj.NClusters), ...
                    int32(obj.Band), ...
                    obj.Metric, ...
                    int32(obj.MaxIter));

                if cost < bestCost
                    bestCost = cost;
                    bestLabels = labels;
                    bestMedoids = medoids;
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
end
