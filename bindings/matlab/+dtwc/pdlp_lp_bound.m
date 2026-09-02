%> @file pdlp_lp_bound.m
%> @brief PDLP first-order solve of the p-median LP relaxation.
%> @author Volkan Kumtepeli
function result = pdlp_lp_bound(D, k, varargin)
%PDLP_LP_BOUND Lower bound on the p-median optimum from its LP relaxation.
%
%   result = dtwc.pdlp_lp_bound(D, k)
%   result = dtwc.pdlp_lp_bound(D, k, 'variant','pdlp', 'tol',1e-8, ...
%                                     'iteration_limit',0, 'use_gpu',false, ...
%                                     'verbose',false)
%
%   Same name, arguments and result fields as the C++ dtwc::mip::pdlp_lp_bound
%   and the Python binding.
%
%   Parameters
%   ----------
%   D : N x N numeric distance matrix (symmetric, zero diagonal).
%   k : number of medoids, 1 <= k <= N.
%   variant : 'pdlp' (cuPDLP-C) or 'hipdlp' (HiGHS PDHG). Default 'pdlp'.
%   tol : PDLP KKT tolerance. Default 1e-8.
%   iteration_limit : PDLP iteration cap; 0 keeps the HiGHS default.
%   use_gpu : ask for the GPU backend (a compile-time property of HiGHS;
%             never a silent downgrade -- warns when unavailable).
%   verbose : let HiGHS print its solver log.
%
%   Returns
%   -------
%   result : struct with fields
%       lp_bound   - LP-relaxation optimum in RAW distance units.
%       solved     - true iff HiGHS reported optimality.
%       iterations - PDLP iterations run.
%       gpu_used   - whether the solve ran on the GPU backend.
%
%   Requires a HiGHS-enabled build; raises 'dtwc:solverError' otherwise.
%
%   See also dtwc.pdlp_gpu_available, dtwc.cluster

    validateattributes(D, {'numeric'}, {'2d', 'square', 'nonempty'}, ...
                       'pdlp_lp_bound', 'D');
    validateattributes(k, {'numeric'}, {'scalar', 'positive', 'integer'}, ...
                       'pdlp_lp_bound', 'k');
    result = dtwc_mex('pdlp_lp_bound', double(D), double(k), varargin{:});
end
