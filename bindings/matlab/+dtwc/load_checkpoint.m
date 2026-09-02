%> @file load_checkpoint.m
%> @brief Load a Problem distance-matrix checkpoint (api-contract-2.0.md §2.7).
%> @author Volkan Kumtepeli
function ok = load_checkpoint(prob, path, metric)
%LOAD_CHECKPOINT Restore a distance matrix into the Problem from a checkpoint dir.
%
%   ok = dtwc.load_checkpoint(prob, dirpath)
%   ok = dtwc.load_checkpoint(prob, dirpath, 'squared_euclidean')
%
%   Returns true if the checkpoint was found and loaded, false otherwise.
%
%   metric is the pointwise metric THIS run computes with: 'l1' (default) or
%   'squared_euclidean' ('sqeuclidean'). A checkpoint written under a different
%   metric no longer matches the identity fingerprint and is rejected (returns
%   false). Mirrors C++ load_checkpoint(prob, path, metric).
%
%   See also dtwc.save_checkpoint, dtwc.CheckpointOptions
    if nargin < 3, metric = 'l1'; end
    ok = dtwc_mex('load_checkpoint', prob.get_handle(), char(path), char(metric));
end
