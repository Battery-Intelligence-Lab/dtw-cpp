%> @file save_binary_checkpoint.m
%> @brief Save a clustering result to a binary checkpoint (api-contract-2.0.md §2.7).
%> @author Volkan Kumtepeli
function save_binary_checkpoint(result, path)
%SAVE_BINARY_CHECKPOINT Save a clustering-result struct to a compact binary file.
%
%   dtwc.save_binary_checkpoint(result, filepath)
%
%   RESULT is a struct with fields labels, medoid_indices, total_cost,
%   iterations, converged (as returned by dtwc.fast_pam / dtwc.clarans / ...).
%   1-based indices are converted to 0-based at the MEX boundary.
%
%   See also dtwc.load_binary_checkpoint
    dtwc_mex('save_binary_checkpoint', result, char(path));
end
