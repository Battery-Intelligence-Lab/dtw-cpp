%> @file CheckpointOptions.m
%> @brief Checkpoint options struct (api-contract-2.0.md §2.7).
%> @author Volkan Kumtepeli
function opts = CheckpointOptions(varargin)
%CHECKPOINTOPTIONS Build a checkpoint-options struct.
%
%   opts = dtwc.CheckpointOptions()
%   opts = dtwc.CheckpointOptions('directory','./checkpoints', ...
%                                 'save_interval',100, 'enabled',true)
%
%   Mirrors dtwc::CheckpointOptions {directory, save_interval, enabled}.
%
%   See also dtwc.save_checkpoint, dtwc.load_checkpoint
    p = inputParser;
    addParameter(p, 'directory', './checkpoints', @(v) ischar(v) || isstring(v));
    addParameter(p, 'save_interval', 100, @(v) isnumeric(v) && isscalar(v));
    addParameter(p, 'enabled', false, @(v) islogical(v) || (isnumeric(v) && isscalar(v)));
    parse(p, varargin{:});

    opts = struct('directory', char(p.Results.directory), ...
                  'save_interval', double(p.Results.save_interval), ...
                  'enabled', logical(p.Results.enabled));
end
