function D = compute_distance_matrix(X, varargin)
%COMPUTE_DISTANCE_MATRIX Compute pairwise DTW distance matrix using DTWC++.
%
%   D = dtwc.compute_distance_matrix(X)
%   D = dtwc.compute_distance_matrix(X, 'Band', 10)
%
%   Parameters
%   ----------
%   X : double matrix (N x L)
%       Each row is a time series of length L.
%   Band : int, optional (default -1)
%       Sakoe-Chiba band width. Use -1 for full DTW.
%
%   Returns
%   -------
%   D : double matrix (N x N)
%       Symmetric pairwise DTW distance matrix.
%
%   Examples
%   --------
%       X = rand(50, 100);  % 50 series of length 100
%       D = dtwc.compute_distance_matrix(X, 'Band', 5);
%
%   See also dtwc.dtw_distance, dtwc.DTWClustering

    p = inputParser;
    addRequired(p, 'X', @(v) isnumeric(v) && ismatrix(v));
    addParameter(p, 'Band', -1, @(v) isnumeric(v) && isscalar(v));
    parse(p, X, varargin{:});

    D = dtwc_mex('compute_distance_matrix', ...
                  double(p.Results.X), ...
                  double(p.Results.Band));
end
