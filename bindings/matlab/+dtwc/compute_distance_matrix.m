%> @file compute_distance_matrix.m
%> @brief Compute pairwise DTW distance matrix using DTWC++.
%> @author Volkan Kumtepeli
function D = compute_distance_matrix(X, varargin)
%COMPUTE_DISTANCE_MATRIX Compute pairwise DTW distance matrix using DTWC++.
%
%   D = dtwc.compute_distance_matrix(X)
%   D = dtwc.compute_distance_matrix(X, 'Band', 10)
%   D = dtwc.compute_distance_matrix(X, 'MatrixDtype', 'float32')
%
%   Parameters
%   ----------
%   X : N x L numeric matrix (a series per row), or a cell of numeric vectors
%       (one series each, any lengths).
%   Band : int, optional (default -1)
%       Sakoe-Chiba band width. Use -1 for full DTW.
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
% @author Volkan Kumtepeli

    p = inputParser;
    addRequired(p, 'X');
    addParameter(p, 'Band', -1, @(v) isnumeric(v) && isscalar(v));
    addParameter(p, 'MatrixDtype', 'float64', @(v) ischar(v) || isstring(v));
    parse(p, X, varargin{:});

    prob = dtwc.Problem('compute_distance_matrix');
    prob.set_matrix_dtype(p.Results.MatrixDtype);
    prob.set_band(p.Results.Band);
    prob.set_data(p.Results.X);
    prob.fill_distance_matrix();
    D = prob.distance_matrix();
end
