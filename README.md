# DTW-C++

DTW-C++ is a dynamic time warping (DTW) and clustering library, written in C++, for time series data. The user can input multiple time series (potentially of variable lengths), and the number of desired clusters (if known), or a range of possible cluster numbers (if the specific number is not known). DTW-C++ can cluster time series data using k-medoids or mixed integer programming (MIP); k-medoids is generally quicker, but may be subject getting stuck in local optima, whereas MIP can find globally optimal clusters.
