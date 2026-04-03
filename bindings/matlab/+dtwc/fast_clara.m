%> @file fast_clara.m
%> @brief Run FastCLARA: scalable k-medoids via subsampling + FastPAM.
%> @author Volkan Kumtepeli
function result = fast_clara(prob, k, varargin)
%FAST_CLARA Run FastCLARA: scalable k-medoids via subsampling + FastPAM.
%
%   result = dtwc.fast_clara(prob, k)
%   result = dtwc.fast_clara(prob, k, 'SampleSize', 100, 'NSamples', 10)
%
%   CLARA avoids O(N^2) memory by running PAM on random subsamples,
%   then assigning all points to the best medoids found.
%
%   Parameters
%   ----------
%   prob : dtwc.Problem
%       Problem with data loaded.
%   k : positive integer
%       Number of clusters.
%   SampleSize : int, optional (default -1 = auto)
%       Subsample size. -1 uses auto formula: max(40+2k, min(N, 10k+100)).
%   NSamples : int, optional (default 5)
%       Number of subsamples to try.
%   MaxIter : int, optional (default 100)
%       Max PAM iterations per subsample.
%   Seed : int, optional (default 42)
%       Random seed for reproducibility.
%
%   Returns
%   -------
%   result : struct (same fields as dtwc.fast_pam)
%
%   Reference: Schubert & Rousseeuw (2021)
%
%   See also dtwc.fast_pam, dtwc.clarans

    p = inputParser;
    addRequired(p, 'prob');
    addRequired(p, 'k', @(v) isnumeric(v) && isscalar(v) && v > 0);
    addParameter(p, 'SampleSize', -1, @(v) isnumeric(v) && isscalar(v));
    addParameter(p, 'NSamples', 5, @(v) isnumeric(v) && isscalar(v) && v > 0);
    addParameter(p, 'MaxIter', 100, @(v) isnumeric(v) && isscalar(v) && v > 0);
    addParameter(p, 'Seed', 42, @(v) isnumeric(v) && isscalar(v));
    parse(p, prob, k, varargin{:});

    result = dtwc_mex('fast_clara', prob.get_handle(), ...
        double(k), ...
        double(p.Results.SampleSize), ...
        double(p.Results.NSamples), ...
        double(p.Results.MaxIter), ...
        double(p.Results.Seed));
end
