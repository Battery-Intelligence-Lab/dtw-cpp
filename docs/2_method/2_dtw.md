---
layout: default
title: Dynamic Time Warping
nav_order: 2
---

# Dynamic Time Warping

The potential approaches for time series clustering can be broadly defined as using a distance metric on the raw data (distance-based), or extracting features or models from the raw data and then clustering. Distance-based methods have many advantages. The most significant advantage is that using the raw data means the results are not biased as can be the case in methods using inputs extracted from the data, because the features or models extracted have to be chosen prior to the clustering process. However, there are also potential disadvantages. Primarily, an incorrect choice of distance-metric can lead to non-logical clusters and picking a correct distance metric can be a very complex task.

Dynamic time warping (DTW) was chosen as the most appropriate distance metric due to it's ability to handle different length inputs and robustness against time shifts, ensuring usage events don't have to occur at the same timestamp for their similarity to be recognised. In some instances this can be disadvantageous if the time of occurance is important for your data. Therefore consideration of the desired output is important.

## DTW Algorithm

Assuming two time series of differing lengths, where:

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

## Speed of Calculation

While there are other clustering algorithms available that handle time-series data with DTW, they are very slow and only allow short data series. DTWpp has been written specifically to quickly handle larger data series. This signficiant speed increase allows the whole DTW matrix to be calculated and then a global optimum for the clustering process can be found (more details in [Clustering](../2_method/3_mip.html)). Other time series clustering packages use k-means clustering which does not garuntee to find a global optimum. This being said, DTW is still a computationally expensive distance metric ($$O(nm)$$ and very long data series may not be suitable. 

### Warping Window

For longer time series it is possible to speed up the DTW calculation by using a 'wapring window'. This works by restricting which data elements on one seies can be mapped to another based on their proximity. For example, if you have two data series of length 100, and a warping window of 10, only elements with a maximum time shift of 10 between the series can be mapped to each other. So, $$y_{1}$$ can only by mapped to $$y_{1}$$ to $$y_{11}$$. The stricter the wapring window, the greater the increase in speed. However, the data being used must be carefully considered to assertain if this will negatively impact the results.

