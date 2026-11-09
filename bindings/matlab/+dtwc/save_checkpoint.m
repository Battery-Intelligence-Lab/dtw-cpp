function save_checkpoint(prob, path, metric)
%SAVE_CHECKPOINT Save the Problem's distance matrix to a checkpoint directory.
%
%   dtwc.save_checkpoint(prob, dirpath)
%   dtwc.save_checkpoint(prob, dirpath, 'squared_euclidean')
%
%   Writes DIRPATH/<name>.dtwm (DIRPATH is created if absent; <name> is the
%   Problem's name, "distances" when empty), replacing any earlier file whole, so
%   a long distance-matrix build can be resumed later via dtwc.load_checkpoint.
%
%   metric is the pointwise metric the stored distances were computed with:
%   'l1' or 'squared_euclidean' ('sqeuclidean'). It is part of the identity
%   fingerprint, so a SquaredL2 matrix is not accepted by a later L1 load.
%   Omitted, it is the Problem's own metric, as C++ save_checkpoint(prob, path)
%   and the CLI take it; name it only for a matrix filled outside the Problem.
%
%   See also dtwc.load_checkpoint, dtwc.CheckpointOptions

%> @file save_checkpoint.m
%> @brief Save a Problem distance-matrix checkpoint.
%> @author Volkan Kumtepeli
    if nargin < 3, metric = ''; end % the MEX takes the Problem's own metric
    dtwc_mex('save_checkpoint', prob.get_handle(), char(path), char(metric));
end
