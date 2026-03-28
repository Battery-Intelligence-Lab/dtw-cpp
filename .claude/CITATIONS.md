# DTWC++ Citations

References used during development. Verify each citation independently before publishing.

---

## DTW and Time Series

- Sakoe, H. & Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 26(1), 43-49.
- Marteau, P.-F. (2009). Time warp edit distances with stiffness adjustment for time series matching. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 31(2), 306-318.
- Jain, B. J. (2018). Semi-Metrification of the Dynamic Time Warping Distance. arXiv:1808.09964.
- Yurtman, A., Soenen, J., Meert, W., & Blockeel, H. (2023). Estimating DTW Distance Between Time Series with Missing Data. *ECML-PKDD 2023*, LNCS 14173.

## Lower Bounds and Fast DTW

- Keogh, E. & Ratanamahatana, C. A. (2005). Exact Indexing of Dynamic Time Warping. *Knowledge and Information Systems*, 7(3), 358-386.
- Kim, S.-W., Park, S., & Chu, W. W. (2001). An Index-Based Approach for Similarity Search Supporting Time Warping in Large Sequence Databases. *ICDE 2001*, 607-614.
- Rakthanmanon, T. et al. (2012). Searching and Mining Trillions of Time Series Subsequences under Dynamic Time Warping. *ACM SIGKDD*, 262-270.
- Lemire, D. (2009). Faster retrieval with a two-pass dynamic-time-warping lower bound. *Pattern Recognition*, 42(9), 2169-2180.

## DTW Variants

- Keogh, E. & Pazzani, M. (2001). Derivative Dynamic Time Warping. *SIAM SDM 2001*.
- Jeong, Y.-S., Jeong, M. K., & Omitaomu, O. A. (2011). Weighted dynamic time warping for time series classification. *Pattern Recognition*, 44(9), 2231-2240.
- Cuturi, M. & Blondel, M. (2017). Soft-DTW: A Differentiable Loss Function for Time-Series. *ICML*, PMLR 70, 894-903.
- Itakura, F. (1975). Minimum Prediction Residual Principle Applied to Speech Recognition. *IEEE TASSP*, 23(1), 67-72.

## Clustering and k-Medoids

- Kaufman, L. & Rousseeuw, P. J. (1987). Clustering by Means of Medoids. In *Statistical Data Analysis Based on the L1-Norm*, North-Holland, 405-416. — Original PAM paper.
- Kaufman, L. & Rousseeuw, P. J. (1990). *Finding Groups in Data: An Introduction to Cluster Analysis*. Wiley. — PAM (Ch. 2), CLARA (Ch. 3).
- Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. *J. Comput. Appl. Math.*, 20, 53-65.
- Schubert, E. & Rousseeuw, P. J. (2021). Fast and eager k-medoids clustering: O(k) runtime improvement of the PAM, CLARA, and CLARANS algorithms. *JMLR*, 22(1), 4653-4688. — FastPAM.
- Ng, R. T. & Han, J. (2002). CLARANS: A Method for Clustering Objects for Spatial Data Mining. *IEEE TKDE*, 14(5), 1003-1016.
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
