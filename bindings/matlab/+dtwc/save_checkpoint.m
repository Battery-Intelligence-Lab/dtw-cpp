%> @file save_checkpoint.m
%> @brief Save a Problem distance-matrix checkpoint (api-contract-2.0.md §2.7).
%> @author Volkan Kumtepeli
function save_checkpoint(prob, path)
%SAVE_CHECKPOINT Save the Problem's distance matrix to a checkpoint directory.
%
%   dtwc.save_checkpoint(prob, dirpath)
%
%   Writes distances.csv + metadata.txt into DIRPATH (created if absent), so a
%   long distance-matrix build can be resumed later via dtwc.load_checkpoint.
%
%   See also dtwc.load_checkpoint, dtwc.CheckpointOptions
    dtwc_mex('save_checkpoint', prob.get_handle(), char(path));
end
