## @UCR_dtai.py
# This script is used for testing timings and memory allocation of the DTAIDistance clustering package using k-medoids.
# It reads time series data, performs clustering using KMedoids, and records the time and memory usage for each operation.
#
# @author Rebecca Perriment
# @date 02 Dec 2022
#

import pandas as pd
import glob
import time
from dtaidistance import dtw, clustering
import numpy as np
import tracemalloc

timings_dtai = {}
memory_dtai = {}

# read time series meta data (number of clusters)
meta = pd.read_csv("../data/benchmark/UCR_DataSummary.csv", index_col="Name")

path = "../data/benchmark/UCRArchive_2018/"

# make text file to store timing data
with open("results_dtai.txt", mode="w") as f:
    f.write("DTAI Results \n")

# go through each time series collection and cluster with dtai, recording time
for fname in glob.glob(path + "/*/"):
    name = fname[34:-1]

    df = pd.read_csv(
        path + str(name) + "/" + str(name) + "_TEST.tsv",
        sep="\t",
        header=None,
        index_col=0,
    )
    n_clusters = meta.loc[name]["Class"]

    # read inital centroids
    init_centroids_df = pd.read_csv(
        "../data/benchmark/UCR_centroids/init_centroids.csv", usecols=[name]
    )
    init_centroids = [
        int(x) for x in init_centroids_df[name].to_list() if str(x) != "nan"
    ]

    df_np = df.to_numpy()

    #tracemalloc.start()  # memory tracking - removed for fair comparison between packages
    t = time.time()  # time tracking
    
    dtai_res = clustering.KMedoids(
        dtw.distance_matrix_fast, {}, k=n_clusters, initial_medoids=init_centroids
    ).fit(list(df_np))

    t2 = time.time()
    timings_dtai[name] = t2 - t

    # memory_dtai[name] = tracemalloc.get_traced_memory()[1]
    # tracemalloc.stop()

    towrite = str("{0}, {1}, {2}").format(name, timings_dtai[name], memory_dtai[name])
    with open("results_dtai.txt", mode="a") as f:
        f.write(towrite)
        f.write("\n")

    print(timings_dtai,end='\n')

