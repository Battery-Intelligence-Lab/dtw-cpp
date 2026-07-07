%> @file device.m
%> @brief Global compute-device get/set (api-contract-2.0.md §1.1).
%> @author Volkan Kumtepeli
function out = device(name)
%DEVICE Get or set the process-wide compute device.
%
%   name = dtwc.device()          % get the current normalised device name
%   name = dtwc.device('cpu')     % set the device; returns the normalised name
%
%   Accepts: 'cpu', 'gpu', 'gpu:N', 'cuda', 'cuda:N', 'hpc' (case-insensitive).
%   Delegates to the single dtwc::Env device registry shared by C++/Python/MATLAB.
%
%   No silent fallback: an unknown name, 'gpu' on a build without a GPU backend,
%   or an 'hpc' .env credential failure raises a 'dtwc:deviceError' — the device
%   is never quietly downgraded to CPU.
%
%   See also dtwc.cluster, dtwc.DTWClustering

    if nargin < 1
        out = dtwc_mex('get_device');
    else
        out = dtwc_mex('set_device', char(name));
    end
end
