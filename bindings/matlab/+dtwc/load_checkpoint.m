%> @file load_checkpoint.m
%> @brief Load a Problem distance-matrix checkpoint (api-contract-2.0.md §2.7).
%> @author Volkan Kumtepeli
function ok = load_checkpoint(prob, path)
%LOAD_CHECKPOINT Restore a distance matrix into the Problem from a checkpoint dir.
%
%   ok = dtwc.load_checkpoint(prob, dirpath)
%
%   Returns true if the checkpoint was found and loaded, false otherwise.
%
%   See also dtwc.save_checkpoint, dtwc.CheckpointOptions
    ok = dtwc_mex('load_checkpoint', prob.get_handle(), char(path));
end
