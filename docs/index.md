---
layout: default
title: Index
nav_exclude: true
---

<!--![](slide_logo.png){:width="80%" }-->


About DTWpp
===========================

_DTWpp_ is a package for time series analysis and clustering. The algorithm ultisies dynamic time warping (DTW) as a distance metric to compare the similarity of input time series. To perform clustering of the time series, mixed integer programming (MIP) is performed on the distance matrix, comparing all time series.

The availability of time series data is rapdily increasing, and analysing and clustering the raw time series data can provide great insights into the data without inflicting biases by extracting features. However clustering time series can become very complex due to their potentially large size, variable lengths and shifts in the time axis. DTW is a powerful distance metric that can compare time series of varying lengths, while allowing shifts in the time axis to recognise similarity between time series even when the events are not at identical time stamps. For further infromation on DTW,  see [Dynamic Time Warping](../5_method/2_dtw.html).

This code has been developed at the Department of Engineering Science of the University of Oxford. 
For information about our battery research, visit the [Battery Intelligence Lab](https://howey.eng.ox.ac.uk) website. 

For more information and comments, please contact 
[david.howey@eng.ox.ac.uk](david.howey@eng.ox.ac.uk).


Requirements
============
You will need a C++ programming environment to edit, compile and run the code.
Visual Studio Code is the environment used to develop the code, but other environments should work as well. Within Visual Studio Code, the extention C++ CMake tools for Windows is required.
Your computer must also have a C++ compiler installed.
The code has been tested using Clang.
Extensive guidelines on how to install those programs is provided in the documentation.

To run the MIP clustering, you will need to have installed MATLAB. 
The code has been tested using MATLAB R2020a, but should work with other releases with no or minor modifications.

 
Installation
============
### Option 1 - Downloading a .zip file ###
[Download a .zip file of the code](https://github.com/Battery-Intelligence-Lab/DTWpp/archive/refs/heads/main.zip)

Then, unzip the folder in a chosen directory on your computer.

### Option 2 - Cloning the repository with Git ###
To clone the repository, you will first need to have Git installed on 
your computer. Then, navigate to the directory where you want to clone the 
repository in a terminal, and type:
```bash
git clone https://github.com/Battery-Intelligence-Lab/DTWpp.git
```
The folder containing all the files should appear in your chosen directory.


Getting started
===============
Detailed instructions on how to get started are in the documentation.
You first have to import the code to your programming environment and make sure the settings are correct (e.g. to allow enough memory for the calculation).
Then you can open main.cpp, which implements the main-function. In this function you choose what to simulate by uncommenting the thing you want to do (and commenting all other lines). 
It is recommended to start with the CCCV-function, which simulates a few CCCV cycles.
You will then have to build (or compile) the code, which might take a while the first time you do this.
Now you can run the code (either locally in the programming environment or by running the executable which was created by the compiler).
While the simulation is running, csv files with the results are written in one or multiple subfolders.
When the simulation has finished, you can run the corresponding MATLAB-script (e.g. readCCCV.m) to plot the outcomes.

Much more detailed documentation can be found in the documentation (from '1 Getting started' to '7 appendixes; debugging, basics of C++, object oriented programming'). These guides are mostly independent of each other, so you don't have to read all of them.
Also the code itself is extensively commented, so you might not have to read the guides at all.


License
=======
This open-source C++ and MATLAB code is published under the BSD 3-clause License,
please read `LICENSE.txt` file for more information.

Two MATLAB functions used by the code to produce the spatial discretisation have been developed by others.
They come with their own licence, see 'license chebdif.txt' and 'licence cumsummat.txt'.
