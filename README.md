DTW-C++
===========================
[![Ubuntu unit](https://github.com/Battery-Intelligence-Lab/dtw-cpp/workflows/Ubuntu%20unit/badge.svg)](https://github.com/Battery-Intelligence-Lab/dtw-cpp/actions)
[![macOS unit](https://github.com/Battery-Intelligence-Lab/dtw-cpp/workflows/macOS%20unit/badge.svg)](https://github.com/Battery-Intelligence-Lab/dtw-cpp/actions)
[![Windows unit](https://github.com/Battery-Intelligence-Lab/dtw-cpp/workflows/Windows%20unit/badge.svg)](https://github.com/Battery-Intelligence-Lab/dtw-cpp/actions)
![Website](https://img.shields.io/website?url=https%3A%2F%2FBattery-Intelligence-Lab.github.io%2Fdtw-cpp%2F)
[![codecov](https://codecov.io/gh/Battery-Intelligence-Lab/dtw-cpp/branch/main/graph/badge.svg?token=K739SRV4QG)](https://codecov.io/gh/Battery-Intelligence-Lab/dtw-cpp)
![Website](https://img.shields.io/website?url=https%3A%2F%2FBattery-Intelligence-Lab.github.io%2Fdtw-cpp%2F)

![Contributors](https://img.shields.io/github/contributors/Battery-Intelligence-Lab/dtw-cpp)
![Last update](https://img.shields.io/github/last-commit/Battery-Intelligence-Lab/dtw-cpp/develop)

![Forks](https://img.shields.io/github/forks/Battery-Intelligence-Lab/dtw-cpp)
![Stars](https://github.com/Battery-Intelligence-Lab/dtw-cpp/develop)

![Contributors](https://img.shields.io/github/contributors/Battery-Intelligence-Lab/dtw-cpp)
![Last update](https://img.shields.io/github/last-commit/Battery-Intelligence-Lab/dtw-cpp/develop)

![Issues](https://img.shields.io/github/issues/Battery-Intelligence-Lab/dtw-cpp)
![Last update](https://img.shields.io/github/last-commit/Battery-Intelligence-Lab/dtw-cpp/develop)

![GitHub all releases](https://img.shields.io/github/downloads/Battery-Intelligence-Lab/dtw-cpp/total) 
[![](https://img.shields.io/badge/license-BSD--3--like-5AC451.svg)](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/LICENSE)

In this `readme.md` a summary is given. You may find the detailed documentation [here](https://Battery-Intelligence-Lab.github.io/dtw-cpp/).  
If you are affected by the sudden change of main branch, please switch to [dtw-cpp_v0.0.2]([https://github.com/Battery-Intelligence-Lab/dtw-cpp/tree/dtw-cpp_v2](https://github.com/Battery-Intelligence-Lab/dtw-cpp/tree/dtwc_0_0_2)) branch. 

Introduction
===========================
DTW-C++ is a dynamic time warping (DTW) and clustering library, written in C++, for time series data. The user can input multiple time series (potentially of variable lengths), and the number of desired clusters (if known), or a range of possible cluster numbers (if the specific number is not known). DTW-C++ can cluster time series data using k-medoids or mixed integer programming (MIP); k-medoids is generally quicker, but may be subject getting stuck in local optima, whereas MIP can find globally optimal clusters.
