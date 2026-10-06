%> @file Result.m
%> @brief Clustering outcome returned by dtwc.cluster (api-contract-2.0.md §1.4).
%> @author Volkan Kumtepeli
classdef Result < handle
%RESULT Clustering outcome (contract §1.4).
%
%   Returned by dtwc.cluster(). It keeps the clustered dtwc.Problem, so
%   score(), save() and distance_matrix() are C++'s (scores::score and the
%   writer dtwc_cl uses): the scores and the four result files, series names
%   included, are the CLI's, byte for byte.
%
%   Properties
%   ----------
%   labels  : double row vector, the 1-based cluster of each series.
%   medoids : double row vector, the 1-based medoid series of each cluster.
%   cost    : double, the total DTW distance of the series to their medoids.
%   device  : char, the device the run computed on ('cpu', 'gpu' or 'gpu:N').
%
%   See also dtwc.cluster, dtwc.silhouette

    properties (SetAccess = private)
        labels
        medoids
        cost
        device
    end

    properties (Access = private)
        Problem   % the clustered dtwc.Problem
    end

    methods
        function obj = Result(prob, result, device)
        %RESULT Made by dtwc.cluster from its Problem and Problem.cluster's result.
            obj.Problem = prob;
            obj.labels = result.labels;
            obj.medoids = result.medoid_indices;
            obj.cost = result.total_cost;
            obj.device = device;
        end

        function s = score(obj, name)
        %SCORE A clustering-quality score by name, computed by C++.
        %   s = res.score('silhouette')        % the MEAN silhouette
        %   s = res.score('davies_bouldin')    % also 'dunn', 'calinski_harabasz', 'inertia'
        %
        %   A matrix-free run (onebatch, clara, tadpole) fills the matrix first.
            s = dtwc_mex('score', obj.Problem.get_handle(), char(name));
        end

        function D = distance_matrix(obj)
        %DISTANCE_MATRIX The N x N DTW distances, filled first after a matrix-free run.
            D = obj.Problem.distance_matrix();
        end

        function save(obj, dir)
        %SAVE Write the four result CSVs into DIR (contract §1.4/§7).
        %   res.save(outdir)
        %
        %   <name>_labels.csv, <name>_medoids.csv, <name>_distance_matrix.csv and
        %   <name>_silhouettes.csv, written by the C++ writer dtwc_cl uses, so the
        %   series names are the dataset's and the bytes are the CLI's.
            dtwc_mex('write_result_files', obj.Problem.get_handle(), char(dir));
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
            title(ax, sprintf('%s: %d clusters (MDS)', obj.Problem.name(), ...
                              numel(obj.medoids)));
            xlabel(ax, 'MDS-1'); ylabel(ax, 'MDS-2');
            if nargout > 0, varargout{1} = ax; end
        end
    end
end
