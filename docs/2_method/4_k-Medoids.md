---
layout: default
title: k-Medoids
nav_order: 4
---

# k-Medoids

k-Medoids is an iterative clustering algorithm that divides the data into $$k$$ groups, each with representative medoids. For time series analysis, the medoids are a single time series in each group, reducing the distance between each time series within the group and itself. The important difference between k-means and k-medoids is that k-medoids uses an actual time series from the dataset rather than a calculated mean, which increases the interpretability of the medoids and eliminates the risk of abstract centroids that do not accurately represent the clusters.

The process of the k-Medoids algorithm is
1. **Select initial medoids.** This can be done either by randomly selecting initial medoids or by using k++ initialisation. k++ initialisation randomly selects the first initial medoid and chooses subsequent initial medoids by selecting time series that are more dissimilar to the original medoid. This spreads the initial medoids apart to ensure a more diverse representation of the data, and reduces the possibility of getting stuck in suboptimal solutions, which is more likely to occur with purely random initialisation.
2. **Cluster assignment.** Calculate the DTW distance between each time series and each of the medoids and then assign each time series to its nearest medoid.
3. **Medoid update.** For each cluster, calculate the cluster dissimilarity (sum of the DTW distances for all time series within a cluster and its medoid) for each time series as the medoid. The time series with the lowest dissimilarity is assigned as the new medoid for the cluster.
4. **Repeat steps 2 and 3 until the medoids do not change.** When the medoids do not change, the algorithm has converged and the k-medoids clustering problem is solved. The user can define a number of iterations at which the algorithm will stop if convergence does not occur.
