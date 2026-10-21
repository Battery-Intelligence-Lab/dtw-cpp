---
title: Getting Started
weight: 1
---

# Getting started

To get started, users should first check the [dependencies](installation.md#dependencies) to ensure they are ready to run the _DTW-C++_ software.

Then users can write a C++ program against the library, starting from the programs in `examples/cpp`, which gives the user more freedom. Or for simpler implementation, the user can use the [command line interface](cli.md).

The format for your input data is detailed [here](supported-data.md). The results are output as .csv files, which the user can specify the location of if desired. The output results include:
- DTW matrix, which contains the DTW distance each all time series with each other.
- Clustering results: `<name>_labels.csv` holds the cluster of each time series and `<name>_medoids.csv` the cluster centers. The total [cost](../method/mip.md) of the clustering problem is printed to the terminal when the run ends; the library's `Problem::write_clusters` writes it at the bottom of its own `.csv`.
- Silhouette score, showing the silouhette score for each time series. The mean of all of these can be considered the total silouhette score for the clustering problem.

Some examples for using the software are detailed [here](examples.md).
