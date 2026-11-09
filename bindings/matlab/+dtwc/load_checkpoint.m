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
%   metric is the pointwise metric THIS run computes with: 'l1' or
%   'squared_euclidean' ('sqeuclidean'). A checkpoint written under a different
%   metric no longer matches the identity fingerprint. Omitted, it is the
%   Problem's own metric, as C++ load_checkpoint(prob, path) and the CLI take
%   it; name it only for a matrix filled outside the Problem.
%
%   See also dtwc.save_checkpoint, dtwc.CheckpointOptions

%> @file load_checkpoint.m
%> @brief Load a Problem distance-matrix checkpoint.
%> @author Volkan Kumtepeli
    if nargin < 3, metric = ''; end % the MEX takes the Problem's own metric
    ok = dtwc_mex('load_checkpoint', prob.get_handle(), char(path), char(metric));
end
