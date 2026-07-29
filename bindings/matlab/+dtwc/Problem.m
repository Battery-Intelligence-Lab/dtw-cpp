%> @file Problem.m
%> @brief OOP wrapper for the DTWC++ Problem class.
%> @author Volkan Kumtepeli
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
%       prob.set_band(10);
%       result = dtwc.fast_pam(prob, 3);
%       sil = dtwc.silhouette(prob);
%
%   See also dtwc.fast_pam, dtwc.silhouette, dtwc.DTWClustering

    properties (Access = private)
        Handle uint64 = uint64(0)
        BandValue double = -1
        VerboseValue logical = false
        MaxIterValue double = 100
        NRepetitionValue double = 1
    end

    properties (Dependent)
        Band double
        Verbose logical
        MaxIter double
        NRepetition double
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

        function set_data(obj, data, names, ndim)
        %SET_DATA Load time series data into the Problem.
        %   prob.set_data(X)                    % X is N x L double matrix (rows = series)
        %   prob.set_data(C)                    % C is a cell array of numeric row vectors
        %                                       %   (ragged / variable-length series)
        %   prob.set_data(X, names)             % names: 1xN cell array of char labels
        %   prob.set_data(X, names, ndim)       % ndim: features per timestep (multivariate,
        %                                       %   interleaved [t0f0 t0f1 t1f0 ...] layout)
        %
        %   Pass names = {} to auto-derive names "0".."N-1".
            if iscell(data)
                % Ragged input: each cell must be a numeric vector. The MEX layer
                % validates class/complexity/shape of every element before use.
                celldata = cellfun(@(v) double(v(:)'), data, 'UniformOutput', false);
                dataArg = celldata;
            else
                validateattributes(data, {'numeric'}, {'2d', 'nonempty'}, 'set_data', 'data');
                dataArg = double(data);
            end

            if nargin < 3, names = {}; end
            if nargin < 4
                dtwc_mex('Problem_set_data', obj.Handle, dataArg, names);
            else
                dtwc_mex('Problem_set_data', obj.Handle, dataArg, names, double(ndim));
            end
        end

        function val = get.Band(obj)
            val = obj.BandValue;
        end

        function set.Band(obj, val)
            warning('dtwc:deprecatedAlias', ...
                ['''dtwc.Problem.Band'' is deprecated; use ' ...
                 '''dtwc.Problem.set_band'' instead.']);
            obj.set_band(val);
        end

        function val = get.Verbose(obj)
            val = obj.VerboseValue;
        end

        function set.Verbose(obj, val)
            warning('dtwc:deprecatedAlias', ...
                ['''dtwc.Problem.Verbose'' is deprecated; use ' ...
                 '''dtwc.Problem.set_verbose'' instead.']);
            obj.set_verbose(val);
        end

        function val = get.MaxIter(obj)
            val = obj.MaxIterValue;
        end

        function set.MaxIter(obj, val)
            warning('dtwc:deprecatedAlias', ...
                ['''dtwc.Problem.MaxIter'' is deprecated; use ' ...
                 '''dtwc.Problem.set_max_iter'' instead.']);
            obj.set_max_iter(val);
        end

        function val = get.NRepetition(obj)
            val = obj.NRepetitionValue;
        end

        function set.NRepetition(obj, val)
            warning('dtwc:deprecatedAlias', ...
                ['''dtwc.Problem.NRepetition'' is deprecated; use ' ...
                 '''dtwc.Problem.set_n_repetitions'' instead.']);
            obj.set_n_repetitions(val);
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
        %   prob.set_distance_strategy('cuda')  % NVIDIA CUDA
        %   prob.set_distance_strategy('metal') % Apple GPU (macOS)
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
            warning('dtwc:deprecatedAlias', ...
                ['''dtwc.Problem.get_distance_matrix'' is deprecated; use ' ...
                 '''dtwc.Problem.distance_matrix'' instead.']);
            D = obj.distance_matrix();
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

        % =================================================================
        %  Canonical 2.0 config setters (snake_case; api-contract-2.0.md §2.1).
        %  The historical PascalCase properties (Band/Verbose/MaxIter/NRepetition)
        %  remain functional as deprecated aliases; these setters are canonical.
        % =================================================================

        function set_band(obj, b)
        %SET_BAND Set the Sakoe-Chiba band (-1 = full DTW). Canonical for `Band`.
            value = double(b);
            if obj.Handle > 0
                dtwc_mex('Problem_set_band', obj.Handle, value);
            end
            obj.BandValue = value;
        end

        function set_verbose(obj, tf)
        %SET_VERBOSE Enable/disable progress messages. Canonical for `Verbose`.
            value = logical(tf);
            if obj.Handle > 0
                dtwc_mex('Problem_set_verbose', obj.Handle, value);
            end
            obj.VerboseValue = value;
        end

        function set_max_iter(obj, n)
        %SET_MAX_ITER Set the maximum iteration count. Canonical for `MaxIter`.
            value = double(n);
            if obj.Handle > 0
                dtwc_mex('Problem_set_max_iter', obj.Handle, value);
            end
            obj.MaxIterValue = value;
        end

        function set_n_repetitions(obj, n)
        %SET_N_REPETITIONS Set the number of random restarts. Canonical for `NRepetition`.
            value = double(n);
            if obj.Handle > 0
                dtwc_mex('Problem_set_n_repetition', obj.Handle, value);
            end
            obj.NRepetitionValue = value;
        end

        function set_method(obj, m)
        %SET_METHOD Set the clustering method ('kmedoids' or 'mip').
            dtwc_mex('Problem_set_method', obj.Handle, char(m));
        end

        function ok = set_solver(obj, s)
        %SET_SOLVER Set the MIP solver ('highs' or 'gurobi').
        %   ok = prob.set_solver('highs')  % ok=false if solver not compiled in
            ok = dtwc_mex('Problem_set_solver', obj.Handle, char(s));
        end

        function set_lb_strategy(obj, s)
        %SET_LB_STRATEGY Set the lower-bound strategy for the pruned CPU path.
        %   'auto' | 'none' | 'kim' | 'keogh' | 'kim_keogh' | 'enhanced' | 'webb'
            dtwc_mex('Problem_set_lb_strategy', obj.Handle, char(s));
        end

        function set_storage_policy(obj, s)
        %SET_STORAGE_POLICY Set storage for the next owning set_data call.
        %   Existing data is unchanged. Values: 'auto' | 'heap' | 'mmap'.
            dtwc_mex('Problem_set_storage_policy', obj.Handle, char(s));
        end

        function set_output_folder(obj, dir)
        %SET_OUTPUT_FOLDER Set the folder where result CSVs are written.
            dtwc_mex('Problem_set_output_folder', obj.Handle, char(dir));
        end

        function set_mip_settings(obj, s)
        %SET_MIP_SETTINGS Configure MIP solver tuning from a struct.
        %   Recognised fields: mip_gap, time_limit_sec, warm_start, numeric_focus,
        %   mip_focus, verbose_solver, max_benders_iter, benders ('auto'|'on'|'off').
            dtwc_mex('Problem_set_mip_settings', obj.Handle, s);
        end

        function s = get_mip_settings(obj)
        %GET_MIP_SETTINGS Return the current MIP settings as a struct.
            s = dtwc_mex('Problem_get_mip_settings', obj.Handle);
        end

        function set_cuda_settings(obj, device_id, precision)
        %SET_CUDA_SETTINGS Configure CUDA dispatch (device_id, precision).
        %   precision: 0 = Auto, 1 = FP32, 2 = FP64.
            if nargin < 3, precision = 0; end
            dtwc_mex('Problem_set_cuda_settings', obj.Handle, double(device_id), double(precision));
        end

        % =================================================================
        %  Canonical 2.0 distance-matrix & clustering methods (§2.2).
        % =================================================================

        function refresh_distance_matrix(obj)
        %REFRESH_DISTANCE_MATRIX Clear the cached distance matrix (recompute on next fill).
            dtwc_mex('Problem_refresh_distance_matrix', obj.Handle);
        end

        function read_distance_matrix(obj, path)
        %READ_DISTANCE_MATRIX Load a distance matrix from a CSV file.
            dtwc_mex('Problem_read_distance_matrix', obj.Handle, char(path));
        end

        function d = max_distance(obj)
        %MAX_DISTANCE Largest entry in the distance matrix.
            d = dtwc_mex('Problem_max_distance', obj.Handle);
        end

        function D = distance_matrix(obj)
        %DISTANCE_MATRIX Get the full NxN distance matrix (canonical for get_distance_matrix).
            D = dtwc_mex('Problem_get_distance_matrix', obj.Handle);
        end

        function cluster(obj)
        %CLUSTER Run the configured clustering method in-place (writes labels/medoids).
            dtwc_mex('Problem_cluster', obj.Handle);
        end

        % =================================================================
        %  Canonical 2.0 read accessors (§2.2). PascalCase dependent props
        %  (Size/ClusterSize/Name/CentroidsInd/ClustersInd) remain as aliases.
        % =================================================================

        function n = size(obj)
        %SIZE Number of time series in the Problem (1-based count).
            n = dtwc_mex('Problem_get_size', obj.Handle);
        end

        function k = n_clusters(obj)
        %N_CLUSTERS Number of clusters currently set on the Problem.
            k = dtwc_mex('Problem_n_clusters', obj.Handle);
        end

        function s = name(obj)
        %NAME Problem name.
            s = dtwc_mex('Problem_get_name', obj.Handle);
        end

        function l = labels(obj)
        %LABELS Cluster assignment per series (int32, 1-based).
            l = dtwc_mex('Problem_get_clusters', obj.Handle);
        end

        function m = medoids(obj)
        %MEDOIDS Medoid series index per cluster (int32, 1-based).
            m = dtwc_mex('Problem_get_centroids', obj.Handle);
        end

        % Dependent property getters
        function val = get.Size(obj)
            warning('dtwc:deprecatedAlias', ...
                ['''dtwc.Problem.Size'' is deprecated; use ' ...
                 '''dtwc.Problem.size'' instead.']);
            val = obj.size();
        end

        function val = get.ClusterSize(obj)
            warning('dtwc:deprecatedAlias', ...
                ['''dtwc.Problem.ClusterSize'' is deprecated; use ' ...
                 '''dtwc.Problem.n_clusters'' instead.']);
            val = obj.n_clusters();
        end

        function val = get.Name(obj)
            warning('dtwc:deprecatedAlias', ...
                ['''dtwc.Problem.Name'' is deprecated; use ' ...
                 '''dtwc.Problem.name'' instead.']);
            val = obj.name();
        end

        function val = get.CentroidsInd(obj)
            warning('dtwc:deprecatedAlias', ...
                ['''dtwc.Problem.CentroidsInd'' is deprecated; use ' ...
                 '''dtwc.Problem.medoids'' instead.']);
            val = obj.medoids();
        end

        function val = get.ClustersInd(obj)
            warning('dtwc:deprecatedAlias', ...
                ['''dtwc.Problem.ClustersInd'' is deprecated; use ' ...
                 '''dtwc.Problem.labels'' instead.']);
            val = obj.labels();
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
