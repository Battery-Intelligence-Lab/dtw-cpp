---
title: 'DTW-C++: A C++ software for fast Dynamic Time Wrapping Clustering'
tags:
  - lead-acid batteries
  - degradation
  - solar house systems
  - C++
authors:
  - name: Volkan Kumtepeli
    orcid: 0000-0003-2392-9771
    affiliation: 1
  - name: Rebecca Perriment
    orcid: 0000-0003-2392-9771
    affiliation: 1
  - name: David A. Howey
    orcid: 0000-0002-0620-3955
    affiliation: 1
affiliations:
  - name: Department of Engineering Science, University of Oxford, OX1 3PJ, Oxford, UK
    index: 1
date: 17 Aug 2022
bibliography: paper.bib
---

# Summary

Disclaimer: paper writing is still ongoing; please do not use this version as a reference. 

``DTW-C++`` is a package for time series analysis and clustering. As the availability of time series data across numerous fields continually increases, developing useful software to intepret and understand this data is essential. Clustering is a useful tool to enable interpretability of large datasets, however it is only effective if useful distance metrics are used. Dynamic time wapring (DTW) is a prominant distance metric for time series analysis, due its robustness against shifts in the time axis and ability to handle time series of different lengths. This allows recognition of similarity between time series even when the events are not at identical time stamps, emphasising on the shape of the time series rather than the time of occurance - if time of occurance is important in the clustering problem, the Euclidean distance as used in other pckages is a better choice. For further infromation on DTW, see [Dynamic Time Warping](../docs/2_method/2_dtw.html). DTW notoriously suffers from slow computation due to it's quadratic complexity, previously making it an unsuitable choice for larger datasets. ``DTW-C++`` speeds up the computation of DTW distance, allowing application to longer time series and larger data sets. In addition, ``DTW-C++`` performs clustering of the time series based off the pairwise DTW distances by formulating the clustering problem as a mixed integer programming (MIP) problem. 

# Statement of need

Clustering time series is becoming increasingly popular as data availability increases; however as the data avilability increases, so does the complexity of the clustering problem. Most time series clustering objectives currently depend on dimension reduction techniques or finding features from the time series {Aghabozorgi} which can induce bias into the clustering. Time series clustering applications range from energy to find consumption patterns, to detecting brainactivity in medical applications, to discovering patterns in stock price trends in the fincance industry. ``DTW_C++`` can handle the large time series datasets, working on the raw data rather than reduced dimension data or selected features from the time series, across the various applications.

Speed comparison against tslearn and dtaidistance

MIP is preferable to other DTW clustering packages which use k-based methods for clustering, as k-based methods are suseptible to sticking in local optima.



# Current ``DTW-C++`` functionality



# Mathmatical background

Consider a time series to be a vector of some arbitrary length. Consider that we have $p$ such vectors in total, each possibly differing in length. To find a subset of $k$ clusters within the set of $p$ vectors we must first make $\frac{(^p C_2)}{2}$ pairwise comparisons between all vectors within the total set and find the `similarity' between each pair. This requires a distance metric to be defined, and dynamic time warping uses local warping (stretching or compressing along the time axis) of the elements within each vector to find optimal alignment (i.e., with minimum cost/distance) between each pair of vectors. 


Comparing two short time series $x$ and $y$ of differing lengths $n$ and $m$

$$
x=(x_1, x_2, ..., x_n) \\
y=(y_1, y_2, ..., y_m).
$$

The cost is the sum of the Euclidean distance between each point and the matched point in the other vector. The following constraints must be met:
* The first and last elements of each series must be matched.
* Only unidirectional forward movement through relative time is allowed, i.e.\ $x_1$ may be mapped to $y_2$ but $x_2$ may not be mapped to $y_1$ (monotonicity).
* Each point is mapped to at least one other point, i.e.\ there are no jumps in time (continuity).

Finding the optimal warping arrangement is an optimisation problem that can be solved using dynamic programming, which splits the problem into easier sub-problems and solves them recursively, storing intermediate solutions until the final solution is reached. For each pairwise comparison, an $n$ by $m$ matrix $C_{n\times m}$ is calculated, where each element represents the cumulative cost between series up to the points $x_i$ and $y_j$:

$$
C_{i,j} = (x_i-y_j)^2+\min\begin{cases}
C_{i-1,j-1}\\
C_{i-1,j}\\
C_{i,j-1}
\end{cases}
$$

The final element $C_{n,m}$ is then the total cost which gives the comparison metric between the two series.

The matrix $C$ is calculated for all pairwise comparisons. The total costs (final element) for each pairwise comparison are stored in a separate symmetric matrix, $D_{p\times p}$ where $p$ is the total number of time series in the clustering exercise. In other words, the element $D_{i,j}$ gives the distance between time series $i$ and $j$.

Using this matrix, $D$, the series can be split into $k$ clusters with integer programming. The problem formulation begins with a $1\times p$ binary vector, $B$, defining if each series is a cluster centroid, in other words for the $i$th element of $B$, 

$$
B_i = \begin{cases}
1, \qquad \text {if centroid}\\
0, \qquad \text {otherwise}
\end{cases}
$$

Only $k$ series can be centroids, therefore

$$
\sum_{i=1}^p B_i=k
$$

A binary square matrix $A_{p\times p}$ is then constructed, where $A_{ij}=1$ if time series $j$ is a member of the $i$ th cluster centroid, and 0 otherwise.

The following constraints apply:
* Each time series must be in one and only one cluster 

$$
\sum_{i=1}^pA_{ij}=1  \quad \forall j \in [1,p]
$$

* Only $k$ rows have non-zero values 

$$
A_{ij} \le B_i \quad \forall i,j \in [1,p]
$$

Then the optimisation problem subject to the above-given constraints becomes:

$$
A^\star, B^\star = \min_{A,B} \sum_i \sum_j D_{ij} \times A_{ij} 
$$

After solving this integer program, the non-zero entries of $B$ represent the centroids and the non-zero elements in the corresponding columns in $A$ represent the members of that cluster.


# Future work



# Acknowledgements

We gratefully acknowledge the contributions by [Battery Intelligence Lab](https://howey.eng.ox.ac.uk) members. 


# References

# Notes

JOSS requirements from paper:
* What problem software is designed to solve
* Who is the target audience


Relevant papers:
* Petitjean
* Frind dtaidistance paper
* Shakoe & Chiba 1978 Dynamic programming algorithm optimization for spokenword recognition

Important points:
* Solving the dtw problem alongside clustering problem allows for most effective parallelization increasing speed – task level parallelization
* Made the problem memory efficient by using vector instead of matrix

