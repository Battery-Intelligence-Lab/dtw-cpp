%> @file Result.m
%> @brief Clustering outcome returned by dtwc.cluster (api-contract-2.0.md §1.4).
%> @author Volkan Kumtepeli
classdef Result < handle
%RESULT Clustering outcome (contract §1.4).
%
%   Returned by dtwc.cluster(). Carries the cluster assignment, medoid indices,
%   total cost and the device the run used, plus score()/save()/plot() helpers.
%   score() and save() are the C++ dtwc::Result members, so the scores and the
%   four output CSVs (series names included) are byte-for-byte the CLI's.
%
%   Properties
%   ----------
%   labels  : int32 row vector (1-based cluster assignment per series).
%   medoids : int32 row vector (1-based medoid series index per cluster).
%   cost    : double, sum of intra-cluster DTW distances.
%   device  : char, normalised device name the run used ('cpu'/'gpu'/'hpc').
%
%   See also dtwc.cluster, dtwc.silhouette

    properties (SetAccess = private)
        labels
        medoids
        cost
        device
    end

    properties (Access = private)
        Handle uint64 = uint64(0)  % C++ dtwc::Result (owns the clustered Problem)
        DatasetName char = 'dataset'
        Dmat = []                  % cached distance matrix for plot()
    end

    methods
        function obj = Result(info)
        %RESULT Construct a Result from the tier1_cluster gateway struct.
            obj.Handle      = info.handle;
            obj.labels      = info.labels;
            obj.medoids     = info.medoid_indices;
            obj.cost        = info.total_cost;
            obj.device      = info.device;
            obj.DatasetName = info.name;
        end

        function delete(obj)
        %DELETE Release the C++ Result (and the Problem it owns).
            if obj.Handle > 0
                try
                    dtwc_mex('Result_delete', obj.Handle);
                catch
                    % MEX may be unloaded during MATLAB shutdown
                end
                obj.Handle = uint64(0);
            end
        end

        function s = score(obj, name)
        %SCORE Evaluate a clustering-quality score by name.
        %   s = res.score('silhouette')        % returns the MEAN silhouette
        %   s = res.score('davies_bouldin')
        %   s = res.score('dunn')
        %   s = res.score('calinski_harabasz')
        %   s = res.score('inertia')
        %
        %   The accepted names and their definitions are C++ Result::score's.
            s = dtwc_mex('Result_score', obj.Handle, char(name));
        end

        function save(obj, dir)
        %SAVE Write the four human-readable result CSVs into DIR (contract §1.4/§7).
        %   res.save(outdir)
        %
        %   Emits <name>_labels.csv, <name>_medoids.csv, <name>_distance_matrix.csv
        %   and <name>_silhouettes.csv through C++ Result::save, so the series
        %   names are the dataset's and the bytes match the CLI's.
            dtwc_mex('Result_save', obj.Handle, char(dir));
        end

        function varargout = plot(obj)
        %PLOT Classical-MDS 2D scatter of the distance matrix, coloured by cluster.
        %   res.plot()          % Python/MATLAB only; C++ writes CSV via save()
        %
        %   Uses a manual classical MDS (double-centred squared-distance eigen-
        %   decomposition) so no toolbox dependency is required.
            D = obj.distance_matrix();
            n = size(D, 1);
            J = eye(n) - ones(n) / n;
            B = -0.5 * (J * (D.^2) * J);
            B = (B + B') / 2;                 % symmetrise against round-off
            [V, E] = eig(B);
            [ev, idx] = sort(diag(E), 'descend');
            ev = max(ev, 0);
            coords = V(:, idx(1:min(2, n))) * diag(sqrt(ev(1:min(2, n))));
            if size(coords, 2) < 2, coords(:, 2) = 0; end

            f = figure('Visible', 'off');
            ax = axes('Parent', f);
            scatter(ax, coords(:, 1), coords(:, 2), 36, double(obj.labels), 'filled');
            title(ax, sprintf('%s: %d clusters (MDS)', obj.DatasetName, ...
                              numel(obj.medoids)));
            xlabel(ax, 'MDS-1'); ylabel(ax, 'MDS-2');
            if nargout > 0, varargout{1} = ax; end
        end
    end

    methods (Access = private)
        function D = distance_matrix(obj)
        %DISTANCE_MATRIX Distance matrix of the clustered Problem, cached.
        %   dtwc::Result owns its Problem privately and publishes the matrix
        %   only through save() (api.hpp), so the matrix is read back from a
        %   scratch save rather than recomputed.
            if isempty(obj.Dmat)
                scratch = [tempname '_dtwc_mds'];
                cleaner = onCleanup(@() remove_directory(scratch));
                dtwc_mex('Result_save', obj.Handle, scratch);
                obj.Dmat = readmatrix( ...
                    fullfile(scratch, [obj.DatasetName '_distance_matrix.csv']), ...
                    'Delimiter', ',', 'NumHeaderLines', 0);
            end
            D = obj.Dmat;
        end
    end
end

function remove_directory(d)
    if exist(d, 'dir')
        rmdir(d, 's');
    end
end
