---
layout: default
title: Clustering
nav_order: 3
---


# Clusteing Method

Mixed integer programming (MIP) can be used on a completed DTW distance matrix to cluster the data series. The output provides both the members of each cluster, where the number of clusters, $$k$$, is determined by the user. Each cluster is represented by a centroid, which is the member of the cluster the reduces the overall cost.

The DTW distance matrix is a square matrix $$D_{n\times n}$$ where $$n$$ is the number of data series in the problem, so $$D_{ij}=C(i,j)$$. The problem formulation begins with a binary square matrix $$A_{n\times n}$$ where $$A_{ij}=1$$ if data series $$j$$ is in the cluster with centroid $$i$$ and 0 otherwise. $$B$$ is a $$1\times n$$ binary vector where

$$
B_{i} = \begin{cases}
    1, \qquad \text {if centroid}\\
    0, \qquad \text {otherwise}
    \end{cases}
$$

$$\sum_{i=1}^n B_{i}=k$$

The following constraints apply:
1. Each data series must be in 1 cluster 

$$ \sum_{i=1}^nA_{ij}=1$$

2. Only $$k$$ rows have non-zero values

$$ A_{ij} \le B_i $$

With the cost function to be minimised:

$$ F=\min \sum_{i} \sum_{j} D_{ij} \odot A_{ij}$$

Where $$\odot$$ represents element-wise multiplication.

After the problem formulation, there are many many possible solutions to be explored. To reduce this linear programming relaxation and branch and bound are used. The relaxation drops the binary constraint, allowing values between 0 and 1, for example a data series could be 0.2 in one cluster, 0.2 in another an 0.6 in another. Relaxing the problem will give a better solution, because there are more degrees of freedom. The cost value here is the lower bound. Each value in $$A$$ is rounded to 0 or 1 and the cost calculated to give a feasible solution. When exploring each branch with relaxation, if any cost is greater than the previously calculated feasible solution, that branch can be cut and not explored anymore because even with relaxation which gives a better cost, it's still greater and therefore a better solution is not possible down that branch. This process continues until the best solution is found. This was implemented using the YALMIP package in MatLab.