# Papers & Sources to Download

Sources referenced in `.claude/UNIMODULAR.md` and `.claude/MISSING.md` that could not be verified via web fetch. If you can download any of these, please place them in a `papers/` folder or provide the PDFs.

---

## Unimodularity & LP Relaxation (UNIMODULAR.md)

### Critical (directly cited, need verification)

0. **Ghouila-Houri, A. (1962)** - "Caracterisation des matrices totalement unimodulaires"
   - Comptes Rendus de l'Academie des Sciences 254, 1192-1194
   - Cited for: necessary and sufficient condition for TU via row partitioning (Section 2.7)
   - Note: this is the foundational theorem used in the rigorous TU proof

0b. **Seymour, P.D. (1980)** - "Decomposition of regular matroids"
   - Journal of Combinatorial Theory, Series B, 28(3), 305-359
   - Cited for: decomposition of TU matrices into network components (Section 2.10)

0c. **Schrijver, A. (1986)** - "Theory of Linear and Integer Programming"
   - Wiley (textbook)
   - Standard reference for TU theory, sufficient conditions

1. **Duran-Mateluna, Ales, Elloumi (2023)** - "Efficient Benders decomposition for the p-median problem"
   - Cited for: scalable Benders decomposition solving p-median with 238k+ points
   - Search: EJOR / European Journal of Operational Research, ~2023
   - Note: was incorrectly cited as "Gendron and Crainic (2022)" in earlier draft

2. **Baiou & Barahona (2009)** - "On the integrality of some facility location polytopes"
   - SIAM J. Discrete Mathematics 23(2), 665-679
   - Cited for: half-integrality of LP extreme points

3. **Baiou & Barahona (2011)** - "On the p-median polytope and the intersection property"
   - SIAM J. Discrete Mathematics 25(1), 1-20
   - Cited for: graph-theoretic LP=IP characterization

4. **Awasthi et al. (2015)** - "Relax, no need to round: integrality of clustering formulations"
   - ITCS 2015
   - Cited for: LP exact recovery under cluster separation

5. **Cohen-Addad et al. (2022)** - "Improved approximation for k-median"
   - STOC 2022
   - Cited for: current best 2.675+eps integrality gap upper bound

6. **Li & Svensson (2016)** - "Approximating k-median via pseudo-approximation"
   - SIAM J. Computing 45(2), 530-547
   - Cited for: 2.732 approximation ratio

7. **Caprara & Fischetti (1996)** - "{0, 1/2}-Chvatal-Gomory cuts"
   - Mathematical Programming 74, 221-235
   - Cited for: polynomial separation of half-integral cuts for p-median

8. **Avella & Sassano (2001)** - "On the p-median polytope"
   - Mathematical Programming 89, 395-411
   - Cited for: facet-defining inequalities

### Well-known (likely correct but good to verify)

9. **Kariv & Hakimi (1979)** - "An algorithmic approach to network location problems"
   - SIAM J. Applied Mathematics 37(3), 539-560
   - NP-hardness of p-median

10. **Cornuejols, Nemhauser, Wolsey (1990)** - "The uncapacitated facility location problem"
    - In Discrete Location Theory, Wiley, 119-171

11. **Charikar, Guha, Tardos, Shmoys (1999)** - "A constant-factor approximation algorithm for the k-median problem"
    - STOC 1999

12. **Jain & Vazirani (2001)** - "Approximation algorithms for metric facility location and k-median problems"
    - JACM 48(2), 274-296

---

## DTW with Missing Data (MISSING.md)

### Critical (directly cited, need verification)

13. **Yurtman, Soenen, Meert, Blockeel (2023)** - "Estimating Dynamic Time Warping Distance Between Time Series with Missing Data"
    - ECML PKDD 2023, LNCS 14173
    - Code: https://github.com/aras-y/DTW_with_missing_values
    - Cited for: DTW-AROW and DTW-CAI algorithms

14. **Tormene, Giorgino, Quaglini, Stefanelli (2009)** - "Matching Incomplete Time Series with Dynamic Time Warping"
    - Artificial Intelligence in Medicine, 45, 11-34
    - Cited for: open-end DTW

15. **Wang & Koniusz (2022)** - "Uncertainty-DTW for Time Series and Sequences"
    - ECCV 2022, arXiv:2211.00005
    - Cited for: uncertainty-aware DTW

16. **Mikalsen et al. (2018)** - "Time Series Cluster Kernel for Learning Similarities Between Multivariate Time Series with Missing Data"
    - Pattern Recognition, 76, 569-581
    - Cited for: TCK kernel approach

17. **Jiang & Arias-Castro (2021)** - "On the Consistency of Metric and Non-Metric K-medoids"
    - AISTATS 2021 (PMLR 130:2485-2493)
    - Cited for: k-medoids consistency for non-metric dissimilarities

18. **Phan et al. (2017)** - "Dynamic Time Warping-Based Imputation for Univariate Time Series Data"
    - Pattern Recognition Letters, 100, 1-7
    - Cited for: DTWBI imputation method

### DTW Metric Properties

19. **Marteau (2009)** - "Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching"
    - IEEE TPAMI 31(2), 306-318
    - Cited for: DTW's failure of triangle inequality, TWED as metric alternative

20. **Jain, B.J. (2018)** - "Semi-Metrification of the Dynamic Time Warping Distance"
    - arXiv:1808.09964
    - Cited for: DTW semi-metric conversion

### Lower Bounds & Clustering Surveys

21. **Holder, Middlehurst, Bagnall (2024)** - "A Review and Evaluation of Elastic Distance Functions for Time Series Clustering"
    - Knowledge and Information Systems, 66, 765-809

22. **Webb (2021)** - "Tight Lower Bounds for Dynamic Time Warping"
    - arXiv:2102.07076

---

## HiGHS / Solver Documentation

23. **HiGHS documentation** - Lazy constraint callbacks, MIP start API
    - https://ergo-code.github.io/HiGHS/dev/
    - Specifically: `Highs::setSolution()` for MIP start, `kCallbackMipDefineLazyConstraints`

24. **Gurobi documentation** - Branching priorities, MIP start, lazy constraints
    - https://www.gurobi.com/documentation/
    - Specifically: `GRB_IntAttr_BranchPriority`, `GRBVar::set(GRB_DoubleAttr_Start, ...)`, `GRBCallback::addLazy()`
