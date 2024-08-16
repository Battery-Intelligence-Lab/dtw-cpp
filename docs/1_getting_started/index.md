---
layout: default
title: Getting started
nav_order: 1
---

# Getting started

To get started, users should first check the [dependencies](1_installation.md#dependencies) to ensure they are ready to run the _DTW-C++_ software.

Then users can choose to run the code by editing `main.cpp`, using a source-code editor such as Visual Studio Code, which gives the user more freedom. Or for simpler implementation, the user can use the [command line interface](2_cli.md).

The format for your input data is detailed [here](3_supported_data.md). The results are output as .csv files, which the user can specify the location of if desired. The output results include:
- DTW matrix, which contains the DTW distance each all time series with each other.
- Clustering results, which contains the cluster centers and the time series belonging to each cluster. The total [cost](../2_method/3_mip.md) of the clustering problem is included at tht bottom on this `.csv`.
- Silhouette score, showing the silouhette score for each time series. The mean of all of these can be considered the total silouhette score for the clustering problem.

Some examples for using the software are detailed [here](4_examples.md).
