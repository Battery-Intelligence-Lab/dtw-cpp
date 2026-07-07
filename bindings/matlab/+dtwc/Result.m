%> @file Result.m
%> @brief Clustering outcome returned by dtwc.cluster (api-contract-2.0.md §1.4).
%> @author Volkan Kumtepeli
classdef Result < handle
%RESULT Clustering outcome (contract §1.4).
%
%   Returned by dtwc.cluster(). Carries the cluster assignment, medoid indices,
%   total cost and the device the run used, plus score()/save()/plot() helpers.
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
        Prob   % dtwc.Problem kept alive so score() can read clustering state
    end

    methods
        function obj = Result(prob, labels, medoids, cost, device)
        %RESULT Construct a Result (called by dtwc.cluster).
            obj.Prob    = prob;
            obj.labels  = labels;
            obj.medoids = medoids;
            obj.cost    = cost;
            obj.device  = device;
        end

        function s = score(obj, name)
        %SCORE Evaluate a clustering-quality score by name.
        %   s = res.score('silhouette')        % returns the MEAN silhouette
        %   s = res.score('davies_bouldin')
        %   s = res.score('dunn')
        %   s = res.score('calinski_harabasz')
        %   s = res.score('inertia')
            nm = lower(strrep(char(name), '-', '_'));
            switch nm
                case 'silhouette'
                    s = mean(dtwc.silhouette(obj.Prob));
                case 'davies_bouldin'
                    s = dtwc.davies_bouldin(obj.Prob);
                case 'dunn'
                    s = dtwc.dunn(obj.Prob);
                case 'calinski_harabasz'
                    s = dtwc.calinski_harabasz(obj.Prob);
                case 'inertia'
                    s = dtwc.inertia(obj.Prob);
                otherwise
                    error('dtwc:invalidArgument', ...
                          ['Unknown score ''%s''. Valid: silhouette, davies_bouldin, ' ...
                           'dunn, calinski_harabasz, inertia.'], char(name));
            end
        end

        function save(obj, dir)
        %SAVE Write the four human-readable result CSVs into DIR (contract §1.4/§7).
        %   res.save(outdir)
        %
        %   Emits <name>_labels.csv, <name>_medoids.csv, <name>_distance_matrix.csv
        %   and <name>_silhouettes.csv, matching the CLI output-file contract.
            if ~exist(dir, 'dir'); mkdir(dir); end
            nm  = obj.Prob.name();
            lab = double(obj.labels);
            med = double(obj.medoids);
            N   = numel(lab);

            % <name>_labels.csv : "name,cluster" (0-based series names + 0-based cluster)
            fid = fopen(fullfile(dir, [nm '_labels.csv']), 'w');
            fprintf(fid, 'name,cluster\n');
            for i = 1:N
                fprintf(fid, '%d,%d\n', i - 1, lab(i) - 1);
            end
            fclose(fid);

            % <name>_medoids.csv : "cluster,medoid_index,medoid_name"
            fid = fopen(fullfile(dir, [nm '_medoids.csv']), 'w');
            fprintf(fid, 'cluster,medoid_index,medoid_name\n');
            for c = 1:numel(med)
                fprintf(fid, '%d,%d,%d\n', c - 1, med(c) - 1, med(c) - 1);
            end
            fclose(fid);

            % <name>_distance_matrix.csv
            D = obj.Prob.distance_matrix();
            writematrix(D, fullfile(dir, [nm '_distance_matrix.csv']));

            % <name>_silhouettes.csv : "name,cluster,silhouette"
            sil = dtwc.silhouette(obj.Prob);
            fid = fopen(fullfile(dir, [nm '_silhouettes.csv']), 'w');
            fprintf(fid, 'name,cluster,silhouette\n');
            for i = 1:N
                fprintf(fid, '%d,%d,%.10g\n', i - 1, lab(i) - 1, sil(i));
            end
            fclose(fid);
        end

        function varargout = plot(obj)
        %PLOT Classical-MDS 2D scatter of the distance matrix, coloured by cluster.
        %   res.plot()          % Python/MATLAB only; C++ writes CSV via save()
        %
        %   Uses a manual classical MDS (double-centred squared-distance eigen-
        %   decomposition) so no toolbox dependency is required.
            D = obj.Prob.distance_matrix();
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
            title(ax, sprintf('%s: %d clusters (MDS)', obj.Prob.name(), numel(obj.medoids)));
            xlabel(ax, 'MDS-1'); ylabel(ax, 'MDS-2');
            if nargout > 0, varargout{1} = ax; end
        end
    end
end
