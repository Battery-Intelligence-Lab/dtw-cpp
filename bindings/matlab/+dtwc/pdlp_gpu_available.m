%> @file pdlp_gpu_available.m
%> @brief Report whether HiGHS was built with the CUDA PDLP backend.
%> @author Volkan Kumtepeli
function tf = pdlp_gpu_available()
%PDLP_GPU_AVAILABLE True when HiGHS was built with CUPDLP_GPU.
%
%   tf = dtwc.pdlp_gpu_available()
%
%   The PDLP compute device is a COMPILE-TIME property of the linked HiGHS
%   build, not a per-call toggle. Same name and meaning as the C++
%   dtwc::mip::pdlp_gpu_available() and the Python binding.
%
%   See also dtwc.pdlp_lp_bound

    tf = dtwc_mex('pdlp_gpu_available');
end
