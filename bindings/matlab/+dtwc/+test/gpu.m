%> @file gpu.m
%> @brief GPU-backend self-probe (parity with dtwc::test::gpu / dtwcpp.test.gpu).
%> @author Volkan Kumtepeli
function report = gpu()
%GPU Execute a tiny GPU kernel and validate it against a CPU oracle.
%
%   report = dtwc.test.gpu()
%
%   Returns a struct whose fields carry the SAME names in C++, Python and
%   MATLAB:
%     available   - logical: a GPU backend is compiled in AND a device is present
%     backend     - char: 'cuda' / 'metal' / '' (none)
%     device_name - char: human-readable device string ('' when unavailable)
%     validated   - logical: GPU kernel matched the CPU oracle within tolerance
%     pass        - logical: available && validated
%     reason      - char: non-empty explanation when ~available or ~validated
%
%   When no GPU backend is compiled in (or no device is present), `available`
%   is false and `reason` names exactly what is missing. Never errors, never
%   silently degrades (2.0 no-silent-fallback rule).
%
%   See also dtwc.test.parallelisation, dtwc.check_system

    report = dtwc_mex('test_gpu');
end
