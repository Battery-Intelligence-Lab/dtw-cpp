%> @file gpu_available.m
%> @brief Whether this build's GPU backend finds a GPU.
%> @author Volkan Kumtepeli
function tf = gpu_available()
%GPU_AVAILABLE True when this build's GPU backend (CUDA, else Metal) finds a GPU.
%
%   tf = dtwc.gpu_available()
%
%   When it is true, dtwc.device('gpu') and Problem.set_device('gpu') compute
%   on that GPU; dtwc.gpu_info() names it.
%
%   See also dtwc.gpu_info, dtwc.device, dtwc.test.gpu

    tf = dtwc_mex('gpu_available');
end
