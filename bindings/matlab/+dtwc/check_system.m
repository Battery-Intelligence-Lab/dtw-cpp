%> @file check_system.m
%> @brief Print a diagnostic summary of available DTWC++ backends.
%> @author Volkan Kumtepeli
function check_system()
%CHECK_SYSTEM Print a diagnostic summary of available DTWC++ backends.
%
%   dtwc.check_system()
%
%   Reports availability of OpenMP (parallel), CUDA (GPU), and MPI
%   (distributed) backends with installation instructions if missing.
%
%   Example
%   -------
%       dtwc.check_system()
%       %  DTWC++ System Check
%       %  ========================================
%       %    ✅ OpenMP: 20 threads
%       %    ❌ CUDA:   not compiled
%       %    ❌ MPI:    not compiled
%       %  ========================================
%
%   See also dtwc.Problem, dtwc.dtw_distance

    info = dtwc_mex('system_check');

    fprintf('DTWC++ System Check\n');
    fprintf('========================================\n');

    % OpenMP
    if info.openmp
        fprintf('  %s OpenMP: %d threads\n', char(9989), info.openmp_threads);
    else
        fprintf('  %s OpenMP: not available\n', char(10060));
        fprintf('     Rebuild with OpenMP support.\n');
        fprintf('     MSVC: /openmp flag (default ON)\n');
        fprintf('     GCC/Clang: -fopenmp\n');
    end

    % CUDA
    if info.cuda
        fprintf('  %s CUDA:   %s\n', char(9989), info.cuda_info);
    else
        if contains(info.cuda_info, 'not compiled')
            fprintf('  %s CUDA:   not compiled\n', char(10060));
            fprintf('     Rebuild: cmake -DDTWC_ENABLE_CUDA=ON ...\n');
        else
            fprintf('  %s CUDA:   compiled but no GPU detected\n', char(10060));
            fprintf('     Check nvidia-smi and CUDA driver.\n');
        end
    end

    % Metal (Apple GPU)
    if isfield(info, 'metal')
        if info.metal
            fprintf('  %s Metal:  %s\n', char(9989), info.metal_info);
        else
            if contains(info.metal_info, 'not compiled')
                fprintf('  %s Metal:  not compiled (macOS only)\n', char(10060));
            else
                fprintf('  %s Metal:  compiled but no GPU detected\n', char(10060));
            end
        end
    end

    % MPI
    if info.mpi
        fprintf('  %s MPI:    available\n', char(9989));
    else
        fprintf('  %s MPI:    not compiled\n', char(10060));
        fprintf('     Rebuild: cmake -DDTWC_ENABLE_MPI=ON ...\n');
        fprintf('     Windows: install MS-MPI SDK from microsoft.com/mpi\n');
    end

    fprintf('========================================\n');
end
