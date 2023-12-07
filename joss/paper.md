---
title: 'DTW-C++: A C++ software for fast Dynamic Time Wrapping Clustering'
tags:
  - C++
  - Dynamic time warping
  - Clustering
  - k-Medoids
  - Mixed integer programming
  - Dynamic programming
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

We present an approach for computationally efficient dynamic time warping (DTW) and clustering of time-series data. The method frames the dynamic warping of time series datasets as an optimisation problem solved using dynamic programming, and then clusters time series data by solving a second optimisation problem using mixed-integer programming (MIP). There is also an option to use k-medoids clustering for increased speed, when a certificate for global optimality is not essential. The improved efficiency of our approach is due to task-level parallelisation of the clustering alongside DTW. Our approach was tested using the UCR Time Series Archive, and was found to be, on average, 33% faster than the next fastest option when using the same clustering method. This increases to 64% faster when considering only larger datasets (with more than 1000 time series). The MIP clustering is most effective on small numbers of longer time series, because the DTW computation is faster than other approaches, but the clustering problem becomes increasingly computationally expensive as the number of time series to be clustered increases.

# Statement of need

Clustering time series is becoming increasingly popular as data availability increases; however as the data avilability increases, so does the complexity of the clustering problem. Most time series clustering objectives currently depend on dimension reduction techniques or finding features from the time series which can induce bias into the clustering [@Aghabozorgi2015]. Dynamic time warping [@Sakoe1978DynamicRecognition] is a well-known technique for manipulating time series to enable comparisons between datasets, using  local warping (stretching or compressing along the time axis) of the elements within each time series to find an optimal alignment between series. This emphasises the similarity of the shapes of the respective time series, rather than the exact alignment of specific features. Unfortunately, DTW does not scale well in computational speed as the length and number of time series to be compared increases---the computational complexity grows quadratically with the total number of data points. This complexity is a barrier to DTW being widely implemented in time series clustering  [@Rajabi2020ASegmentation]. ``DTW-C++`` is written to handle large time series datasets, working on the raw data rather than reduced dimension data or selected features from the time series, across the various applications. 

While there are other packages available for time series clustering using DTW, namely [@Petitjean2011] and [@meert2020wannesm], ``DTW-C++`` offers signficant imporvements in both speed and memory use, allowing larger datasets to be clustered. This is done by task level parallelisation, allowing multiple pairwise comparsions between time series to be evaluated simulataneously, as well as more efficient memory management by solving the DTW distance using only the preceding vector rather than storing the entire warping matrix. This means that the warping path between each time series is not stored, but this is not required for the clustering process - only the final cost is needed. In addition, MIP is preferable to other DTW clustering packages which use k-based methods for clustering, as k-based methods are suseptible to sticking in local optima. MIP finds the global optimum in most cases, and in the rare event that the global optimum is not found, the gap between the best solution found and the global optimum is given.

Time series clustering applications range from energy to find consumption patterns, to detecting brain activity in medical applications, to discovering patterns in stock price trends in the fincance industry. The target audience for this software can therefore range acorss multiple disciplines, intended for any user with a requirement for time-series clustering.


# Current ``DTW-C++`` functionality

The current functionality of the software is as follows:

* Calculate DTW pairwise distances between time series, using a vector based approach to reduce memory use. There is also the option to use a Sakoe-Chiba band to restrict warping in the DTW distance calculation [@Sakoe1978DynamicRecognition]. This speeds up the computation time as well as being a useful constraint for some time series clustering scenarios (e.g., if an event must occur within a certain time window to be considered similar).
* Produce a distance matrix containing all pairwise comparisons between each time series in the dataset.
* Split all time series into a predefined number of clusters, with a representative centroid time series for each cluster. This can be done using MIP or k-medoids clustering, depending on user choice.
* Output the clustering cost, which is the sum of distances between every time series within each cluster and its cluster centroid.
* Find the silhouette score and elbow score for the clusters in order to aid the user decision on how many clusters, $k$, to include.

# Mathmatical background

Consider a time series to be a vector of some arbitrary length. Consider that we have ($p$) such vectors in total, each possibly differing in length. To find a subset of ($k$) clusters within the set of ($p$) vectors using MIP formulation, we must first make $\frac{1}{2} {p \choose 2}$ pairwise comparisons between all vectors within the total set and find the `similarity' between each pair. In this case, the similarity is defined as the DTW distance. Consider two time series ($x$) and ($y$) of differing lengths ($n$) and ($m$) respectively,

$$
x=(x_1, x_2, ..., x_n)
$$
$$
y=(y_1, y_2, ..., y_m).
$$

The DTW distance is the sum of the Euclidean distance between each point and its matched point(s) in the other vector, as shown in \autoref{fig:warping_signals}. The following constraints must be met: 

1. The first and last elements of each series must be matched.
2. Only unidirectional forward movement through relative time is allowed, i.e., if $x_1$ is mapped to $y_2$ then $x_2$ may not be mapped to
    $y_1$ (monotonicity). 
3. Each point is mapped to at least one other point, i.e., there are no jumps in time (continuity).

![Two time series with DTW pairwise alignment between each element, showing one-to-many mapping properties of DTW (left). Cost matrix $C$ for the two time series, showing the warping path and final DTW cost at $C_{14,13}$ (right). \label{fig:warping_signals}](../media/Merged_document.pdf)

Finding the optimal warping arrangement is an optimisation problem that can be solved using dynamic programming, which splits the problem into easier sub-problems and solves them recursively, storing intermediate solutions until the final solution is reached. To understand the memory-efficient method used in ``DTW-C++``, it is useful to first examine the full-cost matrix solution, as follows. For each pairwise comparison, an ($n$) by ($m$) matrix $C^{n\times m}$ is calculated, where each element represents the cumulative cost between series up to the points $x_i$ and $y_j$:

\begin{equation}
    \label{c}
    c_{i,j} = (x_i-y_j)^2+\min\begin{cases}
    c_{i-1,j-1}\\
    c_{i-1,j}\\
    c_{i,j-1}
    \end{cases}
\end{equation}

The final element $c_{n,m}$ is then the total cost, $C_{x,y}$, which provides the comparison metric between the two series $x$ and $y$. \autoref{fig:warping_signals} shows an example of this cost matrix $C$ and the warping path through it.

For the clustering problem, only this final cost for each pairwise comparison is required; the actual warping path (or mapping of each point in one time series to the other) is superfluous for k-medoids clustering. The memory complexity of the cost matrix $C$ is $O(nm)$, so as the length of the time series increases, the memory required increases greatly. Therefore, significant reductions in memory can be made by not storing the entire $C$ matrix. When the warping path is not required, only a vector containing the previous row for the current step of the dynamic programming sub-problem is required (i.e., the previous three values $c_{i-1,j-1}$, $c_{i-1,j}$, $c_{i,j-1}$), as indicated in \autoref{c}.

The DTW distance $C_{x,y}$ is found for each pairwise comparison. As shown in \ref{fig:c_to_d}, pairwise distances are then stored in a separate symmetric matrix, $D^{p\times p}$, where ($p$) is the total number of time series in the clustering exercise. In other words, the element $d_{i,j}$ gives the distance between time series ($i$) and ($j$).

![The DTW costs of all the pairwise comparisons between time series in the dataset are combined to make a distance matrix $D$. \label{fig:c_to_d}](../media/distance_matrix_formation_vector.pdf)

Using this matrix, ($D$), the time series can be split into ($k$) separate clusters with integer programming. The problem formulation begins with a binary square matrix $A^{p\times p}$, where $A_{ij}=1$ if time series ($j$) is a member of the $i$th cluster centroid, and 0 otherwise, as shown in \autoref{fig:A_matrix}.

![Example output from the clustering process, where an entry of 1 indicates that time series $j$ belongs to cluster with centroid $i$. \label{fig:A_matrix}](../media/cluster_matrix_formation4.svg)

As each centroid has to be in its own cluster, non-zero diagonal entries in  $A$ represent centroids. In summary, the following constraints apply: 

1. Only ($k$) series can be centroids,

$$
\sum_{i=1}^p A_{ii}=k.
$$

2. Each time series must be in one and only one cluster,

$$
\sum_{i=1}^pA_{ij}=1  \quad \forall j \in [1,p].
$$

3. In any row, there can only be non-zero entries if the corresponding diagonal entry is non-zero, so a time series can only be in a cluster where the row corresponds to a centroid time series,

$$
A_{ij} \le A_{ii} \quad \forall i,j \in [1,p].
$$

The optimisation problem to solve, subject to the above constraints, is

\begin{equation}
    A^\star = \min_{A} \sum_i \sum_j D_{ij} \times A_{ij}.
\end{equation}

After solving this integer program, the non-zero diagonal entries of ($A$) represent the centroids, and the non-zero elements in the corresponding columns in ($A$) represent the members of that cluster. In the example in \autoref{fig:A_matrix}, the clusters are time series 1, **2**, 5 and 3, **4** with the bold time series being the centroids.

Finding global optimality can increase the computation time, depending on the number of time series within the dataset and the DTW distances. Therefore, there is also a built-in option to cluster using k-medoids, as used in other packages such as \texttt{DTAIDistance} [@meert2020wannesm]. The k-medoids method is often quicker as it is an iterative approach, however it is subject to getting stuck in local optima. The results in the next section show the timing and memory performance of both MIP clustering and k-medoids clustering using \texttt{DTW-C++} compared to other packages.

# Comparison

We compared our approach with two other DTW clustering packages, \texttt{DTAIDistance} [@Meert2020Dtaidistance] and \texttt{TSlearn} [@Tavenard2020TslearnData]. The datasets used for the comparison are from the UCR Time Series Classification Archive [@Dau2018TheArchive], and consist of 128 time series datasets with up to 16,800 data series of lengths up to 2,844. The full results can be found in the Appendix. Benchmarking against  \texttt{TSlearn}  was stopped after the first 22 datasets because the results were consistently over 20 times slower than \texttt{DTW-C++}. Table \ref{tab:small_table} shows the results for datasets downselected to have a number of time series ($N$) greater than 100 and a length of each time series greater than 500 points. This is because \texttt{DTW-C++} is aimed at larger datasets where the speed improvements are more relevant.

<font size= "2">
  
Table: Computational time comparison of \texttt{DTW-C++} using MIP and k-medoids, vs.\ \texttt{DTAIDistance}, and \texttt{TSlearn}, on datasets in the UCR Time Series Classification Archive where $N>100$ and $L>500$. \label{tab}

|                            | Number of time series | Length of time series | DTW-C++ MIP (s) | DTW-C++ k-Medoids (s) | DTAI Distance (s) | Time decrease (%)  |
|----------------------------|-----------------------|-----------------------|-----------------|-----------------------|-------------------|--------------------|
| CinCECGTorso               | 1380                  | 1639                  | 3008.4          | **1104.2**            | 1955.9            | 44                 |
| Computers                  | 250                   | 720                   | 16.1            | **10.5**              | 12.8              | 18                 |
| Earthquakes                | 139                   | 512                   | 3.2             | **2.4**               | 2.5               | 3                  |
| EOGHorizontalSignal        | 362                   | 1250                  | 81.8            | **27.6**              | 82.9              | 67                 |
| EOGVerticalSignal          | 362                   | 1250                  | 85.9            | **30.2**              | 85.2              | 65                 |
| EthanolLevel               | 500                   | 1751                  | 325.7           | **198.9**             | 302.3             | 34                 |
| HandOutlines               | 370                   | 2709                  | 383.7           | **280.9**             | 415.9             | 32                 |
| Haptics                    | 308                   | 1092                  | 65.5            | **24.0**              | 45.5              | 47                 |
| HouseTwenty                | 119                   | 2000                  | 23.8            | **19.1**              | 22.0              | 13                 |
| InlineSkate                | 550                   | 1882                  | 412.4           | **198.9**             | 423.4             | 53                 |
| InsectEPGRegularTrain      | 249                   | 601                   | 12.3            | **5.6**               | 8.9               | 37                 |
| InsectEPGSmallTrain        | 249                   | 601                   | 11.6            | **5.3**               | 8.9               | 41                 |
| LargeKitchenAppliances     | 375                   | 720                   | 44.6            | **25.6**              | 31.8              | 20                 |
| Mallat                     | 2345                  | 1024                  | 2948.7          | **517.0**             | 2251.3            | 77                 |
| MixedShapesRegularTrain    | 2425                  | 1024                  | 2811.8          | **1221.9**            | 2367.1            | 48                 |
| MixedShapesSmallTrain      | 2425                  | 1024                  | 2793.7          | **934.0**             | 2369.3            | 61                 |
| NonInvasiveFetalECGThorax1 | 1965                  | 750                   | 52599.0         | **128.7**             | 941.9             | 86                 |
| NonInvasiveFetalECGThorax2 | 1965                  | 750                   | 4905.4          | **115.6**             | 951.0             | 88                 |
| Phoneme                    | 1896                  | 1024                  | 46549.0         | **198.4**             | 1560.6            | 87                 |
| PigAirwayPressure          | 208                   | 2000                  | 84.6            | **56.7**              | 73.2              | 23                 |
| PigArtPressure             | 208                   | 2000                  | 78.9            | **41.8**              | 71.1              | 41                 |
| PigCVP                     | 208                   | 2000                  | 73.5            | **51.7**              | 69.5              | 26                 |
| RefrigerationDevices       | 375                   | 720                   | 36.8            | **20.3**              | 28.4              | 28                 |
| ScreenType                 | 375                   | 720                   | 38.6            | **16.1**              | 28.5              | 43                 |
| SemgHandGenderCh2          | 600                   | 1500                  | 335.9           | **315.2**             | 325.4             | 3                  |
| SemgHandMovementCh2        | 450                   | 1500                  | 177.7           | **107.2**             | 181.1             | 41                 |
| SemgHandSubjectCh2         | 450                   | 1500                  | 186.4           | **96.7**              | 177.6             | 46                 |
| ShapesAll                  | 600                   | 512                   | 67.5            | **15.1**              | 44.4              | 66                 |
| SmallKitchenAppliances     | 375                   | 720                   | 41.7            | **23.8**              | 30.1              | 21                 |
| StarLightCurves            | 8236                  | 1024                  | N/A             | **18551.7**           | 27558.1           | 33                 |
| UWaveGestureLibraryAll     | 3582                  | 945                   | N/A             | **1194.6**            | 4436.9            | 73                 |


</font>

# Acknowledgements

We gratefully acknowledge the contributions by [Battery Intelligence Lab](https://howey.eng.ox.ac.uk) members, and thank BBOXX for project funding and access to data. This work was also funded by the UKRI PFER Energy Superhub Oxford demonstrator and the ``Data-driven exploration of the carbon emissions impact of grid energy storage deployment and dispatch'' project (EP/W027321/1).


# References


# Appendix
We include here the full benchmarking comparison between \texttt{DTW-C++} (using k-Medoids and MIP), \texttt{DTAIDistance} and \texttt{TSlearn}. As stated in the main text, benchmarking of the latter was discontinued once it was apparent it was significantly slower on all datasets. Additionally, any datasets with a number of time series greater than 3500 were not included for the MIP clustering as the computation time is significantly longer and MIP is not suitable to solve these clustering problems.

|                                | Number of time series | Length of time series | DTW-C++ MIP (s) | DTW-C++ k-Medoids (s) | DTAIDistance (s) | TSlearn (s) |
|--------------------------------|-----------------------|-----------------------|-----------------|-----------------------|------------------|-------------|
| ACSF1                          | 100                   | 1460                  | 10.3943         | 10.038                | 14.50846         | 389.8806    |
| Adiac                          | 391                   | 176                   | 29.304          | 1.26198               | 3.874026         | 172.4454    |
| AllGestureWiimoteX             | 700                   | Vary                  | 200.949         | 2.08638               | 10.82009         |             |
| AllGestureWiimoteY             | 700                   | Vary                  | 104.812         | 2.73356               | 5.791508         |             |
| AllGestureWiimoteZ             | 700                   | Vary                  | 63.5701         | 1.38316               | 5.431429         |             |
| ArrowHead                      | 175                   | 251                   | 2.27864         | 0.897899              | 0.913173         | 60.84465    |
| Beef                           | 30                    | 470                   | 0.148394        | 0.214371              | 0.180589         | 9.44669     |
| BeetleFly                      | 20                    | 512                   | 0.107265        | 0.127118              | 0.077477         | 13.51861    |
| BirdChicken                    | 20                    | 512                   | 0.121148        | 0.132698              | 0.072962         | 7.080103    |
| BME                            | 150                   | 128                   | 2.54826         | 0.231731              | 0.245758         | 28.88994    |
| Car                            | 60                    | 577                   | 0.764395        | 0.402418              | 0.49119          | 53.99938    |
| CBF                            | 900                   | 128                   | 52.708          | 2.69667               | 7.495611         | 264.24      |
| Chinatown                      | 343                   | 24                    | 9.61945         | 0.136138              | 0.286097         | 12.98259    |
| ChlorineConcentration          | 3840                  | 166                   | 18894.9         | 70.1711               | 201.1307         | 1890.485    |
| CinCECGTorso                   | 1380                  | 1639                  | 3008.41         | 1104.24               | 1955.915         | 28990.66    |
| Coffee                         | 28                    | 286                   | 0.091578        | 0.101924              | 0.058483         | 4.686621    |
| Computers                      | 250                   | 720                   | 16.1162         | 10.5249               | 12.81184         | 860.0778    |
| CricketX                       | 390                   | 300                   | 13.4394         | 1.85138               | 6.000062         | 173.9918    |
| CricketY                       | 390                   | 300                   | 13.0524         | 1.87443               | 5.811014         | 192.1667    |
| CricketZ                       | 390                   | 300                   | 13.9427         | 1.95588               | 5.861976         | 279.3739    |
| Crop                           | 16800                 | 46                    |                 | 77.5763               | 6563.978         | 9618.363    |
| DiatomSizeReduction            | 306                   | 345                   | 11.6206         | 2.72367               | 4.685899         | 227.1111    |
| DistalPhalanxOutlineAgeGroup   | 139                   | 80                    | 0.871482        | 0.114135              | 0.161292         | 5.294581    |
| DistalPhalanxOutlineCorrect    | 276                   | 80                    | 5.43763         | 0.420276              | 0.373589         | 8.474895    |
| DistalPhalanxTW                | 139                   | 80                    | 1.70307         | 0.127389              | 0.140715         | 5.276537    |
| DodgerLoopDay                  | 80                    | 288                   | 0.408889        | 0.242574              | 0.248893         |             |
| DodgerLoopGame                 | 138                   | 288                   | 1.55933         | 0.709293              | 0.759482         |             |
| DodgerLoopWeekend              | 138                   | 288                   | 1.5386          | 0.864304              | 0.763335         |             |
| Earthquakes                    | 139                   | 512                   | 3.17021         | 2.40538               | 2.475174         |             |
| ECG200                         | 100                   | 96                    | 0.56719         | 0.171865              | 0.082707         |             |
| ECG5000                        | 4500                  | 140                   |                 | 53.3946               | 206.1784         |             |
| ECGFiveDays                    | 861                   | 136                   | 366.465         | 4.1557                | 6.770701         |             |
| ElectricDevices                | 7711                  | 96                    |                 | 60.5279               | 408.6165         |             |
| EOGHorizontalSignal            | 362                   | 1250                  | 81.7745         | 27.6169               | 82.88655         |             |
| EOGVerticalSignal              | 362                   | 1250                  | 85.8957         | 30.2248               | 85.22367         |             |
| EthanolLevel                   | 500                   | 1751                  | 325.686         | 198.929               | 302.3411         |             |
| FaceAll                        | 1690                  | 131                   | 2118.18         | 3.9713                | 34.63922         |             |
| FaceFour                       | 88                    | 350                   | 0.668328        | 0.53225               | 0.442447         |             |
| FacesUCR                       | 2050                  | 131                   | 1882.32         | 6.43057               | 47.43581         |             |
| FiftyWords                     | 455                   | 270                   | 21.3023         | 2.50822               | 9.540631         |             |
| Fish                           | 175                   | 463                   | 3.62051         | 1.06903               | 2.695585         |             |
| FordA                          | 1320                  | 500                   | 703.644         | 132.113               | 168.9301         |             |
| FordB                          | 810                   | 500                   | 347.998         | 57.843                | 65.1031          |             |
| FreezerRegularTrain            | 2850                  | 301                   | 735.378         | 190.672               | 300.8929         |             |
| FreezerSmallTrain              | 2850                  | 301                   | 730.621         | 252.548               | 296.348          |             |
| Fungi                          | 186                   | 201                   | 1.99303         | 0.569607              | 0.744395         |             |
| GestureMidAirD1                | 130                   | Vary                  | 1.09427         | 0.313373              | 0.367755         |             |
| GestureMidAirD2                | 130                   | Vary                  | 0.948828        | 0.282304              | 0.364915         |             |
| GestureMidAirD3                | 130                   | Vary                  | 0.777829        | 0.338532              | 0.360999         |             |
| GesturePebbleZ1                | 172                   | Vary                  | 3.99312         | 0.378018              | 0.592169         |             |
| GesturePebbleZ2                | 158                   | Vary                  | 4.81083         | 0.344109              | 0.589917         |             |
| GunPoint                       | 150                   | 150                   | 1.35765         | 0.29607               | 0.341601         |             |
| GunPointAgeSpan                | 316                   | 150                   | 7.86469         | 1.11579               | 1.061542         |             |
| GunPointMaleVersusFemale       | 316                   | 150                   | 7.95605         | 0.745125              | 1.15842          |             |
| GunPointOldVersusYoung         | 315                   | 150                   | 7.77761         | 0.773618              | 1.09943          |             |
| Ham                            | 105                   | 431                   | 1.27085         | 0.972879              | 1.009123         |             |
| HandOutlines                   | 370                   | 2709                  | 383.677         | 280.885               | 415.8791         |             |
| Haptics                        | 308                   | 1092                  | 65.493          | 24.0428               | 45.48866         |             |
| Herring                        | 64                    | 512                   | 0.623086        | 0.533143              | 0.533707         |             |
| HouseTwenty                    | 119                   | 2000                  | 23.7889         | 19.1                  | 22.04339         |             |
| InlineSkate                    | 550                   | 1882                  | 412.36          | 198.895               | 423.3659         |             |
| InsectEPGRegularTrain          | 249                   | 601                   | 12.2515         | 5.62819               | 8.897502         |             |
| InsectEPGSmallTrain            | 249                   | 601                   | 11.6215         | 5.31629               | 8.943162         |             |
| InsectWingbeatSound            | 1980                  | 256                   | 93853.4         | 20.0948               | 117.4865         |             |
| ItalyPowerDemand               | 1029                  | 24                    | 56.7128         | 0.871386              | 2.074309         |             |
| LargeKitchenAppliances         | 375                   | 720                   | 44.5535         | 25.5697               | 31.76365         |             |
| Lightning2                     | 61                    | 637                   | 0.844298        | 0.943195              | 0.77562          |             |
| Lightning7                     | 73                    | 319                   | 0.518613        | 0.245382              | 0.290609         |             |
| Mallat                         | 2345                  | 1024                  | 2948.67         | 516.964               | 2251.27          |             |
| Meat                           | 60                    | 448                   | 0.484361        | 0.268107              | 0.355596         |             |
| MedicalImages                  | 760                   | 99                    | 92.8507         | 1.19697               | 3.664919         |             |
| MelbournePedestrian            | 2439                  | 24                    | 144.153         | 2.36702               | 27.96836         |             |
| MiddlePhalanxOutlineAgeGroup   | 154                   | 80                    | 1.3446          | 0.122859              | 0.152365         |             |
| MiddlePhalanxOutlineCorrect    | 291                   | 80                    | 7.59361         | 0.251781              | 0.404614         |             |
| MiddlePhalanxTW                | 154                   | 80                    | 4.56761         | 0.173091              | 0.172191         |             |
| MixedShapesRegularTrain        | 2425                  | 1024                  | 2811.82         | 1221.93               | 2367.125         |             |
| MixedShapesSmallTrain          | 2425                  | 1024                  | 2793.7          | 934.047               | 2369.322         |             |
| MoteStrain                     | 1252                  | 84                    | 131.375         | 4.10743               | 6.979077         |             |
| NonInvasiveFetalECGThorax1     | 1965                  | 750                   | 52599           | 128.662               | 941.9007         |             |
| NonInvasiveFetalECGThorax2     | 1965                  | 750                   | 4905.43         | 115.551               | 950.9619         |             |
| OliveOil                       | 30                    | 570                   | 0.196972        | 0.406894              | 0.211004         |             |
| OSULeaf                        | 242                   | 427                   | 20.9774         | 2.79601               | 4.513287         |             |
| PhalangesOutlinesCorrect       | 858                   | 80                    | 39.8342         | 1.58711               | 3.595135         |             |
| Phoneme                        | 1896                  | 1024                  | 46549           | 198.36                | 1560.56          |             |
| PickupGestureWiimoteZ          | 50                    | Vary                  | 0.142592        | 0.085126              | 0.072093         |             |
| PigAirwayPressure              | 208                   | 2000                  | 84.5969         | 56.658                | 73.22682         |             |
| PigArtPressure                 | 208                   | 2000                  | 78.9084         | 41.8304               | 71.07334         |             |
| PigCVP                         | 208                   | 2000                  | 73.4538         | 51.724                | 69.45288         |             |
| PLAID                          | 537                   | Vary                  | 35.9025         | 6.0825                | 18.25728         |             |
| Plane                          | 105                   | 144                   | 0.515582        | 0.132734              | 0.201192         |             |
| PowerCons                      | 180                   | 144                   | 4.85703         | 0.368735              | 0.364681         |             |
| ProximalPhalanxOutlineAgeGroup | 205                   | 80                    | 6.29439         | 0.141562              | 0.220177         |             |
| ProximalPhalanxOutlineCorrect  | 291                   | 80                    | 7.70703         | 0.234308              | 0.466041         |             |
| ProximalPhalanxTW              | 205                   | 80                    | 6.66169         | 0.188157              | 0.231558         |             |
| RefrigerationDevices           | 375                   | 720                   | 36.8233         | 20.3085               | 28.38962         |             |
| Rock                           | 50                    | 2844                  | 8.7356          | 7.90198               | 8.94623          |             |
| ScreenType                     | 375                   | 720                   | 38.5828         | 16.1376               | 28.46865         |             |
| SemgHandGenderCh2              | 600                   | 1500                  | 335.918         | 315.171               | 325.3503         |             |
| SemgHandMovementCh2            | 450                   | 1500                  | 177.672         | 107.214               | 181.079          |             |
| SemgHandSubjectCh2             | 450                   | 1500                  | 186.388         | 96.7203               | 177.5783         |             |
| ShakeGestureWiimoteZ           | 50                    | Vary                  | 0.173276        | 0.141293              | 0.076811         |             |
| ShapeletSim                    | 180                   | 500                   | 4.83641         | 3.37486               | 3.450108         |             |
| ShapesAll                      | 600                   | 512                   | 67.5375         | 15.1164               | 44.40931         |             |
| SmallKitchenAppliances         | 375                   | 720                   | 41.7491         | 23.7893               | 30.08536         |             |
| SmoothSubspace                 | 150                   | 15                    | 5.48273         | 0.108177              | 0.100557         |             |
| SonyAIBORobotSurface1          | 601                   | 70                    | 45.182          | 0.763845              | 1.43163          |             |
| SonyAIBORobotSurface2          | 953                   | 65                    | 83.0127         | 1.42315               | 3.08514          |             |
| StarLightCurves                | 8236                  | 1024                  |                 | 18551.7               | 27558.11         |             |
| Strawberry                     | 370                   | 235                   | 16.4157         | 3.40908               | 3.515086         |             |
| SwedishLeaf                    | 625                   | 128                   | 54.7593         | 1.07994               | 4.134455         |             |
| Symbols                        | 995                   | 398                   | 177.772         | 30.3962               | 63.36334         |             |
| SyntheticControl               | 300                   | 60                    | 4.76377         | 0.237314              | 0.409787         |             |
| ToeSegmentation1               | 228                   | 277                   | 15.5281         | 1.31453               | 1.779399         |             |
| ToeSegmentation2               | 130                   | 343                   | 1.71634         | 0.984133              | 0.938638         |             |
| Trace                          | 100                   | 275                   | 0.729685        | 0.298243              | 0.353099         |             |
| TwoLeadECG                     | 1139                  | 82                    | 86.9761         | 2.78576               | 5.642078         |             |
| TwoPatterns                    | 4000                  | 128                   | 15892.5         | 36.3566               | 138.4134         |             |
| UMD                            | 144                   | 150                   | 1.29514         | 0.183408              | 0.26445          |             |
| UWaveGestureLibraryAll         | 3582                  | 945                   |                 | 1194.61               | 4436.893         |             |
| UWaveGestureLibraryX           | 3582                  | 315                   |                 | 122.541               | 524.8729         |             |
| UWaveGestureLibraryY           | 3582                  | 315                   |                 | 113.072               | 532.3661         |             |
| UWaveGestureLibraryZ           | 3582                  | 315                   |                 | 107.301               | 525.2477         |             |
| Wafer                          | 6164                  | 152                   |                 | 178.151               | 406.4461         |             |
| Wine                           | 54                    | 234                   | 0.215293        | 0.145209              | 0.130907         |             |
| WordSynonyms                   | 638                   | 270                   | 1525.07         | 3.00847               | 13.84392         |             |
| Worms                          | 77                    | 900                   | 2.61596         | 1.6201                | 1.9559           |             |
| WormsTwoClass                  | 77                    | 900                   | 2.5413          | 2.05345               | 2.154319         |             |
| Yoga                           | 3000                  | 426                   | 1221.19         | 544.997               | 631.1096         |             |


