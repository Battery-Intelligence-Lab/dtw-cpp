%> @file Result.m
%> @brief Clustering outcome returned by dtwc.cluster.
%> @author Volkan Kumtepeli
classdef Result < handle
%RESULT Clustering outcome.
%
%   Returned by dtwc.cluster(). It keeps the run's C++ dtwc::Result, so score(),
%   save() and distance_matrix() are C++'s (scores::score and the writer dtwc_cl
%   uses): the scores and the result files, series names included, are the
%   CLI's, byte for byte. A run that streamed its series (an Arrow IPC file
%   beyond RamLimit) holds none: its score() and distance_matrix(), so plot(),
%   raise dtwc:invalidArgument, and save() writes its labels and medoids, as
%   dtwc_cl does, and the report.
%
%   Properties
%   ----------
%   labels  : double row vector, the 1-based cluster of each series.
%   medoids : double row vector, the 1-based medoid series of each cluster.
%   cost    : double, the total DTW distance of the series to their medoids.
%   device  : char, the device the run computed on ('cpu', 'gpu' or 'gpu:N').
%   Report  : char, the run report, the text dtwc_cl writes to <name>_report.txt
%             (the run's machine, memory plan, stages and result); '' under
%             NoReport.
%
%   See also dtwc.cluster, dtwc.silhouette

    properties (SetAccess = private)
        labels
        medoids
        cost
        device
        Report
    end

    properties (Access = private)
        Handle uint64 = uint64(0)   % the run's C++ dtwc::Result, in the MEX
        Name                        % the run's name, plot's title
    end

    methods
        function obj = Result(out)
        %RESULT Made by dtwc.cluster from what dtwc_mex('run', ...) returns.
            obj.Handle = out.handle;
            obj.labels = out.labels;
            obj.medoids = out.medoids;
            obj.cost = out.cost;
            obj.device = out.device;
            obj.Name = out.name;
            obj.Report = out.report;
        end

        function delete(obj)
        %DELETE Release the run's C++ Result.
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
        %SCORE A clustering-quality score by name, computed by C++.
        %   s = res.score('silhouette')        % the MEAN silhouette
        %   s = res.score('davies_bouldin')    % also 'dunn', 'calinski_harabasz', 'inertia'
        %
        %   A matrix-free run (onebatch, clara, tadpole) fills the matrix first.
            s = dtwc_mex('Result_score', obj.Handle, char(name));
        end

        function D = distance_matrix(obj)
        %DISTANCE_MATRIX The N x N DTW distances, filled first after a matrix-free
        %   run, at the width the run stored them: single under MatrixDtype 'float32'.
            D = dtwc_mex('Result_distance_matrix', obj.Handle);
        end

        function save(obj, dir)
        %SAVE Write the result files into DIR, then the run report.
        %   res.save(outdir)
        %
        %   <name>_labels.csv, <name>_medoids.csv, <name>_distance_matrix.csv and
        %   <name>_silhouettes.csv, written by the C++ writer dtwc_cl uses (C++
        %   Result::save), so the series names are the dataset's and the bytes are
        %   the CLI's; then the run report, <name>_report.txt, last (none under
        %   NoReport). A streamed run writes its labels and medoids and the report.
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
            title(ax, sprintf('%s: %d clusters (MDS)', obj.Name, numel(obj.medoids)));
            xlabel(ax, 'MDS-1'); ylabel(ax, 'MDS-2');
            if nargout > 0, varargout{1} = ax; end
        end
    end
end
