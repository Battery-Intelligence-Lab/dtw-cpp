%> @file load_binary_checkpoint.m
%> @brief Load a clustering result from a binary checkpoint (api-contract-2.0.md §2.7).
%> @author Volkan Kumtepeli
function result = load_binary_checkpoint(path)
%LOAD_BINARY_CHECKPOINT Load a clustering-result struct from a binary checkpoint.
%
%   result = dtwc.load_binary_checkpoint(filepath)
%
%   Returns a struct with fields labels, medoid_indices, total_cost, iterations,
%   converged (labels/medoid_indices are 1-based). Errors 'dtwc:ioError'-style
%   if the file is missing or has an invalid header.
%
%   See also dtwc.save_binary_checkpoint
    result = dtwc_mex('load_binary_checkpoint', char(path));
end
