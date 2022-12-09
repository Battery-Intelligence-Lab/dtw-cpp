import pandas as pd
import numpy as np
import random
import os

main_path = os.path.abspath(__file__ + "/../../")

random.seed(10)

#read time series meta data (number of clusters)
meta = pd.read_csv(main_path + '/data/benchmark/UCR_DataSummary.csv', index_col='Name')
init_centroids = {}

for name in meta.index:
    
    df = pd.read_csv(main_path + '/data/benchmark/UCRArchive_2018/' + str(name) + '/' + str(name) + '_TEST.tsv', sep='\t', header=None, index_col=0)
    n_clusters = meta.loc[name]['Class']

    #find n_clusters random numbers as starting centroids
    init_centroids[name] = random.sample(range(len(df.index)),n_clusters)
    
pd.DataFrame(dict([(a,pd.Series(b)) for a,b in init_centroids.items()])).to_csv(main_path + '/data/benchmark/UCR_centroids/init_centroids.csv')