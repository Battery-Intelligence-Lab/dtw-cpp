%> @file load_checkpoint.m
%> @brief Load a Problem distance-matrix checkpoint.
%> @author Volkan Kumtepeli
function ok = load_checkpoint(prob, path, metric)
%LOAD_CHECKPOINT Restore a distance matrix into the Problem from a checkpoint dir.
%
%   ok = dtwc.load_checkpoint(prob, dirpath)
%   ok = dtwc.load_checkpoint(prob, dirpath, 'squared_euclidean')
%
%   Returns true if the checkpoint was loaded and false, with the Problem
%   unchanged, if DIRPATH holds no checkpoint file for it. A file for other
%   series or another metric raises 'dtwc:invalidArgument', and a file that is not
%   a whole checkpoint raises 'dtwc:ioError'.
%
%   metric is the pointwise metric THIS run computes with: 'l1' (default) or
%   'squared_euclidean' ('sqeuclidean'). A checkpoint written under a different
%   metric no longer matches the identity fingerprint. Mirrors C++
%   load_checkpoint(prob, path, metric).
%
%   See also dtwc.save_checkpoint, dtwc.CheckpointOptions
    if nargin < 3, metric = 'l1'; end
    ok = dtwc_mex('load_checkpoint', prob.get_handle(), char(path), char(metric));
end
