# DTWC++ Citations

References used during development. Verify each citation independently before publishing.

---

## DTW and Time Series

- Sakoe, H. & Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 26(1), 43-49.
- Marteau, P.-F. (2009). Time warp edit distances with stiffness adjustment for time series matching. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 31(2), 306-318.
- Jain, A. (2018). An improved DTW measure for time series classification. arXiv:1808.09964.
- Yurtman, A., Strickx, M., Meysman, P., & Blockeel, H. (2023). DTW-AROW: Dynamic Time Warping with Absent and Reordered Observations in Time Series. *ECML-PKDD 2023*, LNCS 14173.

## Clustering and k-Medoids

- Kaufman, L. & Rousseeuw, P. J. (1990). *Finding Groups in Data: An Introduction to Cluster Analysis*. Wiley.
- Charikar, M., Guha, S., Tardos, E., & Shmoys, D. B. (2002). A constant-factor approximation algorithm for the k-median problem. *Journal of Computer and System Sciences*, 65(1), 129-149.
- Li, S. & Svensson, O. (2013). Approximating k-median via pseudo-approximation. *STOC 2013*.
- Duran-Mateluna, C., Ales, Z., & Elloumi, S. (2023). An efficient Benders decomposition for the p-median problem. *European Journal of Operational Research*.

## Cross-Language Binding Design

- Andersson, J. A. E., Gillis, J., Horn, G., Rawlings, J. B., & Diehl, M. (2019). CasADi: a software framework for nonlinear optimization and optimal control. *Mathematical Programming Computation*, 11(1), 1-36. — Design philosophy for cross-language API consistency (same class/method names across C++/Python/MATLAB).

## pybind11

- Jakob, W., Rhinelander, J., & Moldovan, D. (2017). pybind11 — Seamless operability between C++11 and Python. https://github.com/pybind/pybind11
- pybind11 documentation on GIL management: https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil
- pybind11 documentation on return value policies: https://pybind11.readthedocs.io/en/stable/advanced/functions.html#return-value-policies

## MATLAB MEX

- MathWorks. "C MEX File Applications." MATLAB Documentation. https://www.mathworks.com/help/matlab/matlab_external/c-mex-file-applications.html
- MathWorks. "mexErrMsgIdAndTxt." — Note: This function calls `longjmp`, which skips C++ stack unwinding / destructors.
- MathWorks. "Interleaved Complex API" (R2018a+). https://www.mathworks.com/help/matlab/matlab_external/matlab-support-for-interleaved-complex.html

## Armadillo

- Sanderson, C. & Curtin, R. (2016). Armadillo: a template-based C++ library for linear algebra. *Journal of Open Source Software*, 1(2), 26.
- Column-major storage matches MATLAB (zero-copy possible); differs from NumPy row-major default.
