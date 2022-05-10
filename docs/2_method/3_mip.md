---
layout: default
title: Clustering
nav_order: 3
---


# Clusteing Method

Mixed integer programming (MIP) can be used on a completed DTW distance matrix to cluster the data series. The output provides both the members of each cluster, where the number of clusters, $$k$$, is determined by the user. Each cluster is represented by a centroid, which is the member of the cluster the reduces the overall cost.

The DTW distance matrix is a square matrix $D_{n\times n}$ where $n$ is the number of data series in the problem, so $$D_{ij}=C(i,j)$$. The problem formulation begins with a binary square matrix $$A_{n\times n}$$ where $$A_{ij}=1$$ if data series $$j$$ is in the cluster with centroid $$i$$ and 0 otherwise. $$B$$ is a $$1\times n$$ binary vector where

$$
B_{i} = \begin{cases}
    1, \qquad \text {if centroid}\\
    0, \qquad \text {otherwise}
    \end{cases}
$$

$$\sum_{i=1}^n B_{i}=k$$

The following constraints apply:
1. Each SHS must be in 1 cluster 

$$ \sum_{i=1}^nA_{ij}=1$$

2. Only $k$ rows have non-zero values

$$ D_{ij} \le B_i $$

With the cost function to be minimised:

$$ F=\min \sum_{i} \sum_{j} D_{ij} \odot A_{ij}$$