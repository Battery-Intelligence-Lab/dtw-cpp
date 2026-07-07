%> @file parallelisation.m
%> @brief OpenMP parallelisation self-probe (parity with dtwc::test::parallelisation
%>        / dtwcpp.test.parallelisation).
%> @author Volkan Kumtepeli
function report = parallelisation()
%PARALLELISATION Run a REAL OpenMP region and report engaged threads.
%
%   report = dtwc.test.parallelisation()
%
%   Returns a struct whose fields carry the SAME names in C++, Python and
%   MATLAB:
%     available       - logical: OpenMP is compiled in and usable
%     max_threads     - omp_get_max_threads() (1 without OpenMP)
%     threads_engaged - DISTINCT thread ids observed in a real parallel region
%     pass            - logical: parallelism genuinely engaged
%     reason          - char: non-empty explanation when ~available
%
%   Proof-of-engagement, not a compile-flag read. On a sequential MEX build
%   (configured with -DDTWC_ALLOW_SEQUENTIAL=ON) `available` is false and
%   `reason` names the sequential build — the honest loud answer, never faked.
%
%   See also dtwc.test.gpu, dtwc.check_system

    report = dtwc_mex('test_parallelisation');
end
