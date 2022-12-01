## Benchmarking Data

The data used for benchmarking this project is the UCR time series data set [REF]. This can be found and downloaded from the UCR Time Series Classification Archive website at: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/

The data should be downloaded and put into the Data -> Benchmarking folder if you would like to run the benchmarking tests. 

**Note** that the format of the UCR dataset when it is donwloaded is not directly compatible with this library. Therefore the user must change the format from the downloaded TSV files into individual CSV files for each time series, as outlined [here](/../dtw-cpp/docs/1_getting_started/2_install).

## Benchmarking Tests
This library was benchmarked against two other DTW clustering packages: [Tslearn](https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.TimeSeriesKMeans.html) and [DTAIDistance](https://dtaidistance.readthedocs.io/en/latest/).