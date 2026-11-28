function save_checkpoint(prob, path)
%SAVE_CHECKPOINT Save the Problem's distance matrix to a checkpoint directory.
%
%   dtwc.save_checkpoint(prob, dirpath)
%
%   Writes DIRPATH/<name>.dtwm (DIRPATH is created if absent; <name> is the
%   Problem's name, "distances" when empty), replacing any earlier file whole, so
%   a long distance-matrix build can be resumed later via dtwc.load_checkpoint.
%
%   The Problem's metric is part of the identity fingerprint, so a SquaredL2
%   matrix is not accepted by a later L1 load.
%
%   See also dtwc.load_checkpoint, dtwc.CheckpointOptions

%> @file save_checkpoint.m
%> @brief Save a Problem distance-matrix checkpoint.
%> @author Volkan Kumtepeli
%> @date 07 Jul 2026
    dtwc_mex('save_checkpoint', prob.get_handle(), char(path));
end
