function D = compute_distance_matrix(X, varargin)
%COMPUTE_DISTANCE_MATRIX Compute pairwise DTW distance matrix using DTWC++.
%
%   D = dtwc.compute_distance_matrix(X)
%   D = dtwc.compute_distance_matrix(X, 'Band', 10)
%   D = dtwc.compute_distance_matrix(X, 'Metric', 'squared_euclidean')
%   D = dtwc.compute_distance_matrix(X, 'Device', 'gpu')
%   D = dtwc.compute_distance_matrix(X, 'MatrixDtype', 'float32')
%
%   Parameters
%   ----------
%   X : N x L numeric matrix (a series per row), or a cell of numeric vectors
%       (one series each, any lengths).
%   Band : int, optional (default -1)
%       Sakoe-Chiba band width. Use -1 for full DTW.
%   Metric : char, optional (default 'l1')
%       Pointwise cost metric: 'l1', 'euclidean', or 'squared_euclidean'.
%   Device : char, optional (default '' -> dtwc.device())
%       Compute device ('cpu', 'gpu', 'gpu:N', 'cuda', 'cuda:N', 'cpu+gpu', 'cpu+gpu:N').
%   MatrixDtype : char, optional (default 'float64')
%       The matrix's value type: 'float64', or 'float32' (half the memory,
%       each distance rounded to a single).
%
%   Returns
%   -------
%   D : double matrix (N x N), single for MatrixDtype 'float32'
%       Symmetric pairwise DTW distance matrix.
%
%   Examples
%   --------
%       X = rand(50, 100);  % 50 series of length 100
%       D = dtwc.compute_distance_matrix(X, 'Band', 5);
%
%   See also dtwc.distance.dtw, dtwc.DTWClustering

%> @file compute_distance_matrix.m
%> @brief Compute pairwise DTW distance matrix using DTWC++.
%> @author Volkan Kumtepeli
%> @date 29 Mar 2026
    p = inputParser;
    addRequired(p, 'X');
    addParameter(p, 'Band', -1, @(v) isnumeric(v) && isscalar(v));
    addParameter(p, 'Metric', 'l1', @(v) ischar(v) || isstring(v));
    addParameter(p, 'Device', '', @(v) isempty(v) || ischar(v) || isstring(v));
    addParameter(p, 'MatrixDtype', 'float64', @(v) ischar(v) || isstring(v));
    parse(p, X, varargin{:});

    dev = char(p.Results.Device);
    if isempty(dev)
        dev = dtwc.device();
    end
    prob = dtwc.Problem('compute_distance_matrix', 'Device', dev);
    cleaner = onCleanup(@() delete(prob)); %#ok<NASGU>
    prob.set_matrix_dtype(p.Results.MatrixDtype);
    prob.set_distance('Band', p.Results.Band, 'Metric', char(p.Results.Metric));
    prob.set_data(p.Results.X);
    D = prob.distance_matrix();
end
