function ok = load_checkpoint(prob, path)
%LOAD_CHECKPOINT Restore a distance matrix into the Problem from a checkpoint dir.
%
%   ok = dtwc.load_checkpoint(prob, dirpath)
%
%   Returns true if the checkpoint was loaded and false, with the Problem
%   unchanged, if DIRPATH holds no checkpoint file for it. A file for other
%   series, or computed under another metric than the Problem's, raises
%   'dtwc:invalidArgument', and a file that is not a whole checkpoint raises
%   'dtwc:ioError'.
%
%   See also dtwc.save_checkpoint, dtwc.CheckpointOptions

%> @file load_checkpoint.m
%> @brief Load a Problem distance-matrix checkpoint.
%> @author Volkan Kumtepeli
    ok = dtwc_mex('load_checkpoint', prob.get_handle(), char(path));
end
