classdef Problem < handle
%PROBLEM OOP wrapper for the DTWC++ Problem class.
%
%   Manages a C++ Problem object via a uint64 handle. Uses MEX calls for
%   all data transfer and computation. Follows the CasADi pattern:
%   create handle -> configure -> run algorithms -> inspect results.
%
%   Example:
%       prob = dtwc.Problem('my_problem');
%       prob.set_data(randn(20, 100));
%       prob.Band = 10;
%       result = dtwc.fast_pam(prob, 3);
%       sil = dtwc.silhouette(prob);
%
%   See also dtwc.fast_pam, dtwc.silhouette, dtwc.DTWClustering

    properties (Access = private)
        Handle uint64 = uint64(0)
    end

    properties
        Band double = -1
        Verbose logical = false
        MaxIter double = 100
        NRepetition double = 1
    end

    properties (SetAccess = private, Dependent)
        Size
        ClusterSize
        Name
        CentroidsInd
        ClustersInd
    end

    methods
        function obj = Problem(name)
        %PROBLEM Create a new DTWC++ Problem object.
        %   prob = dtwc.Problem()
        %   prob = dtwc.Problem('my_problem')
            if nargin < 1, name = ''; end
            obj.Handle = dtwc_mex('Problem_new', name);
            % Sync cached properties from C++ defaults
            obj.Band = -1;
            obj.Verbose = false;
            obj.MaxIter = 100;
            obj.NRepetition = 1;
        end

        function delete(obj)
        %DELETE Release the C++ Problem object.
            if obj.Handle > 0
                try
                    dtwc_mex('Problem_delete', obj.Handle);
                catch
                    % MEX may be unloaded during MATLAB shutdown
                end
                obj.Handle = uint64(0);
            end
        end

        function set_data(obj, data)
        %SET_DATA Load time series data into the Problem.
        %   prob.set_data(X) where X is an N x L double matrix.
        %   Each row is one time series of length L.
            validateattributes(data, {'numeric'}, {'2d', 'nonempty'}, 'set_data', 'data');
            dtwc_mex('Problem_set_data', obj.Handle, double(data));
        end

        function set.Band(obj, val)
            obj.Band = val;
            if obj.Handle > 0
                dtwc_mex('Problem_set_band', obj.Handle, double(val));
            end
        end

        function set.Verbose(obj, val)
            obj.Verbose = val;
            if obj.Handle > 0
                dtwc_mex('Problem_set_verbose', obj.Handle, logical(val));
            end
        end

        function set.MaxIter(obj, val)
            obj.MaxIter = val;
            if obj.Handle > 0
                dtwc_mex('Problem_set_max_iter', obj.Handle, double(val));
            end
        end

        function set.NRepetition(obj, val)
            obj.NRepetition = val;
            if obj.Handle > 0
                dtwc_mex('Problem_set_n_repetition', obj.Handle, double(val));
            end
        end

        function fill_distance_matrix(obj)
        %FILL_DISTANCE_MATRIX Compute all pairwise DTW distances.
        %   prob.fill_distance_matrix()
            dtwc_mex('Problem_fill_distance_matrix', obj.Handle);
        end

        function d = dist_by_ind(obj, i, j)
        %DIST_BY_IND Get DTW distance between series i and j (1-based).
        %   d = prob.dist_by_ind(i, j)
            d = dtwc_mex('Problem_dist_by_ind', obj.Handle, double(i), double(j));
        end

        function set_n_clusters(obj, k)
        %SET_N_CLUSTERS Set the number of clusters.
        %   prob.set_n_clusters(k)
            dtwc_mex('Problem_set_n_clusters', obj.Handle, double(k));
        end

        function set_variant(obj, variant, varargin)
        %SET_VARIANT Set the DTW variant.
        %   prob.set_variant('standard')
        %   prob.set_variant('ddtw')
        %   prob.set_variant('wdtw', 0.1)      % g parameter
        %   prob.set_variant('adtw', 2.0)       % penalty
        %   prob.set_variant('softdtw', 0.5)    % gamma
            if nargin > 2
                dtwc_mex('Problem_set_variant', obj.Handle, variant, double(varargin{1}));
            else
                dtwc_mex('Problem_set_variant', obj.Handle, variant);
            end
        end

        function set_missing_strategy(obj, strategy)
        %SET_MISSING_STRATEGY Set NaN handling strategy.
        %   prob.set_missing_strategy('error')
        %   prob.set_missing_strategy('zero_cost')
        %   prob.set_missing_strategy('arow')
        %   prob.set_missing_strategy('interpolate')
            dtwc_mex('Problem_set_missing_strategy', obj.Handle, strategy);
        end

        function set_distance_strategy(obj, strategy)
        %SET_DISTANCE_STRATEGY Set distance matrix computation strategy.
        %   prob.set_distance_strategy('auto')
        %   prob.set_distance_strategy('brute_force')
        %   prob.set_distance_strategy('pruned')
            dtwc_mex('Problem_set_distance_strategy', obj.Handle, strategy);
        end

        function cost = find_total_cost(obj)
        %FIND_TOTAL_COST Compute total cost of current clustering.
        %   cost = prob.find_total_cost()
            cost = dtwc_mex('Problem_find_total_cost', obj.Handle);
        end

        function D = get_distance_matrix(obj)
        %GET_DISTANCE_MATRIX Get the full NxN distance matrix.
        %   D = prob.get_distance_matrix()
            D = dtwc_mex('Problem_get_distance_matrix', obj.Handle);
        end

        function set_distance_matrix(obj, D)
        %SET_DISTANCE_MATRIX Set a precomputed distance matrix.
        %   prob.set_distance_matrix(D) where D is NxN symmetric double.
            dtwc_mex('Problem_set_distance_matrix', obj.Handle, double(D));
        end

        function filled = is_distance_matrix_filled(obj)
        %IS_DISTANCE_MATRIX_FILLED Check if distance matrix is computed.
        %   filled = prob.is_distance_matrix_filled()
            filled = dtwc_mex('Problem_is_distance_matrix_filled', obj.Handle);
        end

        % Dependent property getters
        function val = get.Size(obj)
            val = dtwc_mex('Problem_get_size', obj.Handle);
        end

        function val = get.ClusterSize(obj)
            val = dtwc_mex('Problem_get_cluster_size', obj.Handle);
        end

        function val = get.Name(obj)
            val = dtwc_mex('Problem_get_name', obj.Handle);
        end

        function val = get.CentroidsInd(obj)
            val = dtwc_mex('Problem_get_centroids', obj.Handle);
        end

        function val = get.ClustersInd(obj)
            val = dtwc_mex('Problem_get_clusters', obj.Handle);
        end

        function disp(obj)
        %DISP Display Problem summary.
            try
                info = dtwc_mex('Problem_get_info', obj.Handle);
                fprintf('  dtwc.Problem: "%s"\n', info.name);
                fprintf('    Size: %d series\n', info.size);
                fprintf('    Band: %d\n', info.band);
                fprintf('    Verbose: %d\n', info.verbose);
                fprintf('    Distance matrix filled: %d\n', info.dist_filled);
            catch
                fprintf('  dtwc.Problem [invalid handle]\n');
            end
        end

        function h = get_handle(obj)
        %GET_HANDLE Return the internal C++ handle (for MEX calls).
            h = obj.Handle;
        end
    end
end
