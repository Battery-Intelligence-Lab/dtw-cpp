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

``DTW-C++`` is a package for time series analysis and clustering. As the availability of time series data across numerous fields continually increases, developing useful software to intepret and understand this data is essential. Clustering is a useful tool to enable interpretability of large datasets, however it is only effective if useful distance metrics are used. Dynamic time wapring (DTW) is a prominant distance metric for time series analysis, due its robustness against shifts in the time axis and ability to handle time series of different lengths. This allows recognition of similarity between time series even when the events are not at identical time stamps, emphasising the shape of the time series rather than the time of occurance - if time of occurance is important in the clustering problem, the Euclidean distance as used in other pckages is a better choice. For further infromation on DTW, see [Dynamic Time Warping](../docs/2_method/2_dtw.html). DTW notoriously suffers from slow computation due to it's quadratic complexity, previously making it an unsuitable choice for larger datasets. ``DTW-C++`` speeds up the computation of DTW distance, allowing application to longer time series and larger data sets. In addition, ``DTW-C++`` performs clustering of the time series based off the pairwise DTW distances by formulating the clustering problem as a mixed integer programming (MIP) problem.

# Statement of need

Clustering time series is becoming increasingly popular as data availability increases; however as the data avilability increases, so does the complexity of the clustering problem. Most time series clustering objectives currently depend on dimension reduction techniques or finding features from the time series which can induce bias into the clustering (@Aghabozorgi2015). Time series clustering applications range from energy to find consumption patterns, to detecting brainactivity in medical applications, to discovering patterns in stock price trends in the fincance industry. ``DTW-C++`` is written to handle large time series datasets, working on the raw data rather than reduced dimension data or selected features from the time series, across the various applications. The target audience for this software can therefore range acorss multiple disciplines, intended for any user with a requirement for time-series clustering.

While there are other packages available for time series clustering using DTW, namely @Petitjean2011 and @meert2020wannesm, ``DTW-C++`` offers signficant imporvements in both speed and memory use, allowing larger datasets to be clustered. This is done by task level parallelisation, allowing multiple pairwise comparsions between time series to be evaluated simulataneously, as well as more efficient memory management by solving the DTW distance using only the preceding vector rather than storing the entire warping matrix. This means that the warping path between each time series is not stored, but this is not required for the clustering process - only the final cost is needed. In addition, MIP is preferable to other DTW clustering packages which use k-based methods for clustering, as k-based methods are suseptible to sticking in local optima. MIP finds the global optimum in most cases, and in the rare event that the global optimum is not found, the gap between the best solution found and the global optimum is given.


# Current ``DTW-C++`` functionality

The main features of ``DTW-C++`` are as follows:

* Load all time series data from CSV files, as detailed in [Dynamic Time Warping](../docs/2_method/2_dtw.html).
* Produce a distance matrix - pairwaise comparsion between each time series in the dataset.
* Option to band the warping potnential of the DTW alignment, as originally detialed in @Sakoe1978. This can speed up the computation time of the DTW cost as well as being a useful constraint for some time series clustering scenarios (e.g. if event must occur within a certian time window to be considered similar).
* Find clusters groupings and centeroids fro a predefined $k$ or a range of $k$ values, using either MIP or k-Medoids, as described below in [Mathmatical background](#mathmatical background).
* Output the final clustering cost, as well as a sillouette score (@Shahapure2020) for each $k$ value if running for multiple $k$ values.

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

![DTW example for two time series showing a) the one-to-many mapping and b) the warping bath between the timer series](https://user-images.githubusercontent.com/93582518/206199528-29489727-e4d3-4067-bcc0-e5ae11c43820.PNG)

The matrix $C$ is calculated for all pairwise comparisons. The total costs (final element) for each pairwise comparison are stored in a separate symmetric matrix, $D_{p\times p}$ where $p$ is the total number of time series in the clustering exercise. In other words, the element $D_{i,j}$ gives the distance between time series $i$ and $j$.

![Distance matrix formation for $p$ time series](https://user-images.githubusercontent.com/93582518/202716790-11704c18-99bc-4234-b5db-3b21940ad91d.PNG)

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

![MIP cluster matrix $A$ formation for an example scenario with 5 time series and 2 clusters. The clusters are time series 1, **2**, 5 and 3, **4** with the bold time series being the centorids.](https://user-images.githubusercontent.com/93582518/206171442-ba6044a5-656a-491f-bb78-98564a0475a1.PNG)

Then the optimisation problem subject to the above-given constraints becomes:

$$
A^\star, B^\star = \min_{A,B} \sum_i \sum_j D_{ij} \times A_{ij} 
$$

After solving this integer program, the non-zero entries of $B$ represent the centroids and the non-zero elements in the corresponding columns in $A$ represent the members of that cluster. In the example in Figure 3, the clusters are time series 1, **2**, 5 and 3, **4** with the bold time series being the centorids.


Finding global optimality can increase the computation time, depending on the number of time series within the dataset and DTW distances. Therefore there is also a built in feature to cluster using k-Medoids, as is used in other packages such as DTAIDistance. k-Medoids is often quicker as it is an iterative method, however it is subject to getting stuck in local optima. The table in [Comparison](#comparison) shows the timing and memory performance of both MIP clustering and k-Medoids clustering cmpared to other packages.

# Comparison

We have compared our library to two other standard DTW clustering packages, DTAIDistance and TSlearn. The datasets used are time series from the UCR Time Series Classification Archive (@UCRArchive2018), cosisting of 128 time series datasets with up to 16,800 data series of lengths up to 2,844.

                               | DTW-C++               || DTAISDistance              || Tslearn  |
Dataset                        | Time (s) | Memory (MB) | Time (s)      | Memory (MB) | Time (s) | Memory (MB)
------------------------------ | -------- | ----------- | ------------- | ----------- | ---------|---------------
ACSF1                          |          |             | 14.51         | 18.14       | 389.88   | 1691.47    
Adiac                          |          |             | 3.87          | 4.34        | 172.45   | 15.58      
ArrowHead                      |          |             | 0.91          | 1.22        | 60.84    | 99.72      
Beef                           |          |             | 0.18          | 0.33        | 9.45     | 45.58      
BeetleFly                      |          |             | 0.08          | 0.11        | 13.52    | 62.41      
BirdChicken                    |          |             | 0.07          | 0.27        | 7.08     | 53.89      
BME                            |          |             | 0.25          | 0.95        | 28.89    | 22.68      
Car                            |          |             | 0.49          | 0.58        | 54.00    | 236.10     
CBF                            |          |             | 7.50          | 18.58       | 264.24   | 103.40     
Chinatown                      |          |             | 0.29          | 3.09        | 12.98    | 2.79       
ChlorineConcentration          |          |             | 201.13        | 305.52      | 1890.48  | 765.07     
CinCECGTorso                   |          |             | 1955.92       | 58.20       | 28990.66 | 21456.77   
Coffee                         |          |             | 0.06          | 0.13        | 4.69     | 26.60      
Computers                      |          |             | 12.81         | 3.04        | 860.08   | 1642.15    
CricketX                       |          |             | 6.00          | 4.70        | 173.99   | 95.34      
CricketY                       |          |             | 5.81          | 4.69        | 192.17   | 119.23     
CricketZ                       |          |             | 5.86          | 4.67        | 279.37   | 137.25     
Crop                           |          |             | 6563.98       | 5675.32     | 9618.36  | 122.47     
DiatomSizeReduction            |          |             | 4.69          | 3.31        | 227.11   | 363.81     
DistalPhalanxOutlineAgeGroup   |          |             | 0.16          | 0.67        | 5.29     | 8.32       
DistalPhalanxOutlineCorrect    |          |             | 0.37          | 2.25        | 8.47     | 21.94      
DistalPhalanxTW                |          |             | 0.14          | 0.68        | 5.28     | 4.01       
Earthquakes                    |          |             | 2.48          | 1.15        |          |            
ECG200                         |          |             | 0.08          | 0.41        |          |            
ECG5000                        |          |             | 206.18        | 416.50      |          |            
ECGFiveDays                    |          |             | 6.77          | 17.18       |          |            
ElectricDevices                |          |             | 408.62        | 1206.34     |          |            
EOGHorizontalSignal            |          |             | 82.89         | 6.77        |          |            
EOGVerticalSignal              |          |             | 85.22         | 6.77        |          |            
EthanolLevel                   |          |             | 302.34        | 12.73       |          |            
FaceAll                        |          |             | 34.64         | 61.34       |          |            
FaceFour                       |          |             | 0.44          | 0.68        |          |            
FacesUCR                       |          |             | 47.44         | 89.14       |          |            
FiftyWords                     |          |             | 9.54          | 5.96        |          |            
Fish                           |          |             | 2.70          | 1.66        |          |            
FordA                          |          |             | 168.93        | 42.00       |          |            
FordB                          |          |             | 65.10         | 17.53       |          |            
FreezerRegularTrain            |          |             | 300.89        | 173.46      |          |            
FreezerSmallTrain              |          |             | 296.35        | 173.64      |          |            
Fungi                          |          |             | 0.74          | 1.42        |          |            
GunPoint                       |          |             | 0.34          | 0.85        |          |            
GunPointAgeSpan                |          |             | 1.06          | 2.98        |          |            
GunPointMaleVersusFemale       |          |             | 1.16          | 2.95        |          |            
GunPointOldVersusYoung         |          |             | 1.10          | 2.96        |          |            
Ham                            |          |             | 1.01          | 0.88        |          |            
HandOutlines                   |          |             | 415.88        | 11.26       |          |            
Haptics                        |          |             | 45.49         | 5.03        |          |            
Herring                        |          |             | 0.53          | 0.59        |          |            
HouseTwenty                    |          |             | 22.04         | 2.35        |          |            
InlineSkate                    |          |             | 423.37        | 15.10       |          |            
InsectEPGRegularTrain          |          |             | 8.90          | 2.80        |          |            
InsectEPGSmallTrain            |          |             | 8.94          | 2.79        |          |            
InsectWingbeatSound            |          |             | 117.49        | 85.31       |          |            
ItalyPowerDemand               |          |             | 2.07          | 22.99       |          |            
LargeKitchenAppliances         |          |             | 31.76         | 5.67        |          |            
Lightning2                     |          |             | 0.78          | 0.62        |          |            
Lightning7                     |          |             | 0.29          | 0.54        |          |            
Mallat                         |          |             | 2251.27       | 132.52      |          |            
Meat                           |          |             | 0.36          | 0.52        |          |            
MedicalImages                  |          |             | 3.66          | 13.40       |          |            
MiddlePhalanxOutlineAgeGroup   |          |             | 0.15          | 0.94        |          |            
MiddlePhalanxOutlineCorrect    |          |             | 0.40          | 2.43        |          |            
MiddlePhalanxTW                |          |             | 0.17          | 0.95        |          |            
MixedShapesRegularTrain        |          |             | 2367.12       | 140.95      |          |            
MixedShapesSmallTrain          |          |             | 2369.32       | 140.96      |          |            
MoteStrain                     |          |             | 6.98          | 34.13       |          |            
NonInvasiveFetalECGThorax1     |          |             | 941.90        | 91.85       |          |            
NonInvasiveFetalECGThorax2     |          |             | 950.96        | 91.87       |          |            
OliveOil                       |          |             | 0.21          | 0.35        |          |            
OSULeaf                        |          |             | 4.51          | 2.33        |          |            
PhalangesOutlinesCorrect       |          |             | 3.60          | 16.64       |          |            
Phoneme                        |          |             | 1560.56       | 90.18       |          |            
PigAirwayPressure              |          |             | 73.23         | 4.52        |          |            
PigArtPressure                 |          |             | 71.07         | 4.51        |          |            
PigCVP                         |          |             | 69.45         | 4.51        |          |            
Plane                          |          |             | 0.20          | 0.49        |          |            
PowerCons                      |          |             | 0.36          | 1.26        |          |            
ProximalPhalanxOutlineAgeGroup |          |             | 0.22          | 1.40        |          |            
ProximalPhalanxOutlineCorrect  |          |             | 0.47          | 2.45        |          |            
ProximalPhalanxTW              |          |             | 0.23          | 1.42        |          |            
RefrigerationDevices           |          |             | 28.39         | 5.51        |          |            
Rock                           |          |             | 8.95          | 1.25        |          |            
ScreenType                     |          |             | 28.47         | 5.51        |          |            
SemgHandGenderCh2              |          |             | 325.35        | 15.27       |          |            
SemgHandMovementCh2            |          |             | 181.08        | 10.09       |          |            
SemgHandSubjectCh2             |          |             | 177.58        | 10.09       |          |            
ShapeletSim                    |          |             | 3.15          | 1.78        |          |            
ShapesAll                      |          |             | 44.41         | 10.54       |          |            
SmallKitchenAppliances         |          |             | 30.09         | 5.66        |          |            
SmoothSubspace                 |          |             | 0.10          | 0.68        |          |            
SonyAIBORobotSurface1          |          |             | 1.43          | 8.57        |          |            
SonyAIBORobotSurface2          |          |             | 3.09          | 20.20       |          |            
StarLightCurves                |          |             | 27558.11      | 1436.21     |          |            
Strawberry                     |          |             | 3.52          | 4.12        |          |            
SwedishLeaf                    |          |             | 4.13          | 9.49        |          |            
Symbols                        |          |             | 63.36         | 24.42       |          |            
SyntheticControl               |          |             | 0.41          | 2.52        |          |            
ToeSegmentation1               |          |             | 1.78          | 2.01        |          |            
ToeSegmentation2               |          |             | 0.94          | 1.03        |          |            
Trace                          |          |             | 0.35          | 0.71        |          |            
TwoLeadECG                     |          |             | 5.64          | 28.47       |          |            
TwoPatterns                    |          |             | 138.41        | 329.76      |          |            
UMD                            |          |             | 0.26          | 0.94        |          |            
UWaveGestureLibraryAll         |          |             | 4436.89       | 288.90      |          |            
UWaveGestureLibraryX           |          |             | 524.87        | 270.86      |          |            
UWaveGestureLibraryY           |          |             | 532.37        | 270.86      |          |            
UWaveGestureLibraryZ           |          |             | 525.25        | 270.86      |          |            
Wafer                          |          |             | 406.45        | 776.29      |          |            
Wine                           |          |             | 0.13          | 0.39        |          |            
WordSynonyms                   |          |             | 13.84         | 10.59       |          |            
Worms                          |          |             | 1.96          | 0.93        |          |            
WormsTwoClass                  |          |             | 1.95          | 0.92        |          |            
Yoga                           |          |             | 631.11        | 194.70      |          |            


# Acknowledgements

We gratefully acknowledge the contributions by [Battery Intelligence Lab](https://howey.eng.ox.ac.uk) members. 

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

# References



