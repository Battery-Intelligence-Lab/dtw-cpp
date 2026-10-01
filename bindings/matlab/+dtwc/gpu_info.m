%> @file gpu_info.m
%> @brief One line on this build's GPU backend and its GPU.
%> @author Volkan Kumtepeli
function info = gpu_info()
%GPU_INFO One line naming this build's GPU backend and the GPU 'gpu' computes on.
%
%   info = dtwc.gpu_info()   % e.g. 'CUDA: NVIDIA RTX 4000 Ada Generation (...)'
%
%   'CUDA: <device>' or 'Metal: <device>', or why there is no GPU.
%
%   See also dtwc.gpu_available, dtwc.device, dtwc.test.gpu

    info = dtwc_mex('gpu_info');
end
