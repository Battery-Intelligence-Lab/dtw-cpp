%> @file save_checkpoint.m
%> @brief Save a Problem distance-matrix checkpoint (api-contract-2.0.md §2.7).
%> @author Volkan Kumtepeli
function save_checkpoint(prob, path, metric)
%SAVE_CHECKPOINT Save the Problem's distance matrix to a checkpoint directory.
%
%   dtwc.save_checkpoint(prob, dirpath)
%   dtwc.save_checkpoint(prob, dirpath, 'squared_euclidean')
%
%   Publishes a new immutable generation under DIRPATH (created if absent), so a
%   long distance-matrix build can be resumed later via dtwc.load_checkpoint.
%
%   metric is the pointwise metric the stored distances were computed with:
%   'l1' (default) or 'squared_euclidean' ('sqeuclidean'). It is part of the
%   identity fingerprint, so a SquaredL2 matrix is not accepted by a later L1
%   load. Mirrors C++ save_checkpoint(prob, path, metric) and the CLI --metric.
%
%   See also dtwc.load_checkpoint, dtwc.CheckpointOptions
    if nargin < 3, metric = 'l1'; end
    dtwc_mex('save_checkpoint', prob.get_handle(), char(path), char(metric));
end
