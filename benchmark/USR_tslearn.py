#code for testing timings and memory allocation of TSlearn clustering package using k-means

import pandas as pd
import glob
import time
from tslearn.clustering import TimeSeriesKMeans
import numpy as np #numbas dependancy of numpy version 1.2 or less
import tracemalloc

timings_ts = {}
memory_ts = {}

#read time series meta data (number of clusters)
meta = pd.read_csv('../data/benchmark/UCR_DataSummary.csv', index_col='Name')

path = '../data/benchmark/UCRArchive_2018/*/'

#make text file to store timing and memory data
with open('results_ts.txt', mode='w') as f:
    f.write('TSlearn Results \n')

#go through each time series collection and cluster with dtai, recording time and memory
for fname in glob.glob(path):

    name = fname[34:-1]

    df = pd.read_csv('../data/benchmark/UCRArchive_2018/' + str(name) + '/' + str(name) + '_TEST.tsv', sep='\t', header=None, index_col=0)
    n_clusters = meta.loc[name]['Class']
 
    tracemalloc.start() #memory tracking
    t = time.time()     #time tracking

    tslearn_res = TimeSeriesKMeans(n_clusters=n_clusters, metric='dtw').fit(df)

    t2 = time.time()
    timings_ts[name] = t2-t

    memory_ts[name] = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    
    towrite = str('{0}, {1}, {2}').format(name, timings_ts[name], memory_ts[name])
    with open('results_dtai.txt', mode='a') as f:
        f.write(towrite)
        f.write('\n')
    
    print(towrite)