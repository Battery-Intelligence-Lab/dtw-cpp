---
layout: default
title: Dynamic Time Warping
nav_order: 2
---

# Dynamic Time Warping

The potential approaches for time series clustering can be broadly defined as using a distance metric on the raw data (distance-based), or extracting features or models from the raw data and then clustering. Distance-based methods have many advantages. The most significant advantage is that using the raw data means the results are not biased as can be the case in methods using inputs extracted from the data, because the features or models extracted have to be chosen prior to the clustering process. However, there are also potential disadvantages. Primarily, an incorrect choice of distance-metric can lead to non-logical clusters and picking a correct distance metric can be a very complex task.

Dynamic time warping (DTW) was chosen as the most appropriate distance metric due to it's ability to handle different length inputs and robustness against time shifts, ensuring usage events don't have to occur at the same timestamp for their similarity to be recognised. In some instances this can be disadvantageous if the time of occurance is important for your data. Therefore consideration of the 

## Speed of Calculation

While there are other clustering algorithms available that handle time-series data with DTW, they are very slow and only allow short data series. DTWpp has been written specifically to quickly handle larger data series. This signficiant speed increase allows the whole DTW matrix to be calculated and then a global optimum for the clustering process can be found (more details in [Clustering](../2_method/3_mip.html)). Other time series clustering packages use k-means clustering which does not garuntee to find a global optimum.

This being said, DTW is still a computationally expensive distance metric and very long data series may not be suitable.

