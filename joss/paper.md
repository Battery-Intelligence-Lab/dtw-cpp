---
title: 'DTW-C++: A C++ software for fast Dynamic Time Wrapping Clustering'
tags:
  - lead-acid batteries
  - degradation
  - solar house systems
  - C++
authors:
  - name: Rebecca Perriment
    orcid: 0000-0003-2392-9771
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Volkan Kumtepeli
    orcid: 0000-0003-2392-9771
    affiliation: 1
  - name: David A. Howey
    orcid: 0000-0002-0620-3955
    affiliation: 1
affiliations:
  - name: Department of Engineering Science, University of Oxford, OX1 3PJ, Oxford, UK
    index: 1
  - name: Becky's second affiliation 
    index: 2
date: 17 Aug 2022
bibliography: paper.bib
---

# Summary

Disclaimer: paper writing is still ongoing; please do not use this version as a reference. 

Day by day, [@reiners2022digital], [@reniers2019review].

``DTW-C++`` (simulator for lithium-ion degradation) is a package for time series analysis and clustering. The algorithm ultisies dynamic time warping (DTW) as a distance metric to compare the similarity of input time series. To perform clustering of the time series, mixed integer programming (MIP) is performed on the distance matrix, comparing all time series.

The availability of time series data is rapdily increasing, and analysing and clustering the raw time series data can provide great insights into the data without inflicting biases by extracting features. However clustering time series can become very complex due to their potentially large size, variable lengths and shifts in the time axis. DTW is a powerful distance metric that can compare time series of varying lengths, while allowing shifts in the time axis to recognise similarity between time series even when the events are not at identical time stamps. For further infromation on DTW, see [Dynamic Time Warping](../docs/2_method/2_dtw.html).

An example reference: [@kumtepeli2020energy].

# Mathematical background

The potential approaches for time series clustering can be broadly defined as using a distance metric on the raw data (distance-based), or extracting features or models from the raw data and then clustering. Distance-based methods have many advantages. The most significant advantage is that using the raw data means the results are not biased as can be the case in methods using inputs extracted from the data, because the features or models extracted have to be chosen prior to the clustering process. However, there are also potential disadvantages. Primarily, an incorrect choice of distance-metric can lead to non-logical clusters and picking a correct distance metric can be a very complex task.

Dynamic time warping (DTW) was chosen as the most appropriate distance metric due to it's ability to handle different length inputs and robustness against time shifts, ensuring usage events don't have to occur at the same timestamp for their similarity to be recognised. In some instances this can be disadvantageous if the time of occurance is important for your data. Therefore consideration of the desired output is important.

$$X=x_{1} + x_{2} + ... + x_{n}$$

$$Y=x_{1} + y_{2} + ... + y_{m}$$

Dynamic programming is used to construct an $$n$$ by $$m$$ matrix where for each element a cumulative cost between the corresponding points $$x_{i}$$ and $$y_{j}$$ is calculated

$$
c(i,j) = (x_i-y_j)^2+\min\begin{cases}
    c(i-1,j-1)\\
    c(i-1,j)\\
    c(i,j-1)
    \end{cases}
$$

The min function allows the warping process to occur. The function finds if it is a lower cost to match the next value in $$Y$$ with the current value in $$X$$ or visa versa, or if the corresponding values of each are the lowest cost. This exemplifies DTWs one-to-many property. It is also important to note the monotonic  and continuity conditions on the warping path. 

$$i_{t-1}\leq i_t \mbox{  and  } j_{t-1}\leq j_t$$

The monotonic condition ensures only unidirectional, forward movement through relative time, i.e. $$x_{1}$$ could be mapped to $$y_{2}$$ but then $$x_{2}$$ could not be mapped to $$y_{1}$$. 

$$i_t-i_{t-1}\leq 1 \mbox{  and  } j_t-j_{t-1}\leq 1$$

The continuity condition ensures each point is mapped to at least one other point so there are no jumps in time.

The min function dictates the optimal warping path through the matrix from $$(1,1)$$ to $$(n,m)$$, with the final DTW cost:

$$ C=c(n,m) $$

# Acknowledgements

We gratefully acknowledge the contributions by [Battery Intelligence Lab](https://howey.eng.ox.ac.uk) members. 


# References

