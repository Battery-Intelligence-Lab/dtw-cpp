---
title: Getting Started
weight: 1
---

# Getting started

To get started, users should first check the [dependencies](installation.md#dependencies) to ensure they are ready to run the _DTW-C++_ software.

Then users can write a C++ program against the library, starting from the programs in `examples/cpp`, which gives the user more freedom. Or for simpler implementation, the user can use the [command line interface](cli.md).

The format for your input data is detailed [here](supported-data.md). The results are output as .csv files, which the user can specify the location of if desired. The output results include:
- DTW matrix, `<name>_distance_matrix.csv`, which contains the DTW distance each all time series with each other. It is written only when the method has filled the matrix: `onebatch`, `tadpole`, `clara` on a sample smaller than N, and a run that streams its series fill none.
- Clustering results: `<name>_labels.csv` holds the cluster of each time series and `<name>_medoids.csv` the cluster centers. The total [cost](../method/mip.md) of the clustering problem is printed to the terminal when the run ends; the library's `Problem::write_clusters` writes it at the bottom of its own `.csv`.
- Silhouette score, `<name>_silhouettes.csv`, showing the silhouette score for each time series (written with the distance matrix, for k > 1). The mean of all of these can be considered the total silhouette score for the clustering problem.
- The run report, `<name>_report.txt`: the run's machine, memory plan, stages and result in plain text. `--no-report` writes none.

Some examples for using the software are detailed [here](examples.md).
