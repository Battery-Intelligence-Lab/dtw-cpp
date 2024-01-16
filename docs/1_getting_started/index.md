---
layout: default
title: Getting started
nav_order: 1
---

# Getting started

To get started, users should first check the [dependencies](dependencies.md) to ensure they are ready to run the _DTW-C++_ software.

Then users can choose to run the code [directly](direct_use.md), using a source-code editor such as Visual Studio Code, which gives the user more freedom. Or for simpler implementation, the user can use the [command line interface](cli.md).

The format for your input data is detailed [here](supported_data.md). The results are output as .csv files, which the user can specify the location of if desired. The output results include:
- DTW matrix, which contains the DTW distance each all time series with each other.
- Clustering results, which contains the cluster centers and the time series belonging to each cluster. The filenames of the input files are used as the labels for each time series. The total [cost]( of the clustering problem 
