%> @file build_dendrogram.m
%> @brief Build a dendrogram from a Problem.
%> @author Volkan Kumtepeli
function dend = build_dendrogram(prob, varargin)
%BUILD_DENDROGRAM Build a dendrogram from a Problem.
%
%   dend = dtwc.build_dendrogram(prob)
%   dend = dtwc.build_dendrogram(prob, 'Linkage', 'complete')
%   dend = dtwc.build_dendrogram(prob, 'Linkage', 'average', 'MaxPoints', 1000)
%
%   Performs agglomerative hierarchical clustering and returns the full
%   dendrogram (N-1 merge steps). Use dtwc.cut_dendrogram to get flat
%   clusters at a specific k.
%
%   Parameters
%   ----------
%   prob : dtwc.Problem
%       Problem with distance matrix filled.
%   Linkage : char, optional (default 'average')
%       Linkage criterion: 'single', 'complete', or 'average'.
%   MaxPoints : int, optional (default 2000)
%       Safety guard: throws if N exceeds this.
%
%   Returns
%   -------
%   dend : struct with fields:
%       merges    - (N-1) x 4 double matrix [cluster_a, cluster_b, distance, new_size]
%                   cluster_a and cluster_b are 1-based.
%       n_points  - int32 scalar
%
%   See also dtwc.cut_dendrogram, dtwc.fast_pam
% @author Volkan Kumtepeli

    p = inputParser;
    addRequired(p, 'prob');
    addParameter(p, 'Linkage', 'average', @ischar);
    addParameter(p, 'MaxPoints', 2000, @(v) isnumeric(v) && isscalar(v) && v > 0);
    parse(p, prob, varargin{:});

    dend = dtwc_mex('build_dendrogram', prob.get_handle(), ...
        p.Results.Linkage, ...
        double(p.Results.MaxPoints));
end
