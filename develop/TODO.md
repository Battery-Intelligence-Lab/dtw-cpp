# Information about this branch of this library: 
- Only a test version, not complete. 

# Disclaimer: 
- This TODO list is pretty much informal text of what is in our mind. 

## TODO: 

### Current priority: 

- [ ] Commenting
- [ ] MATLAB / Python integration
- [ ] JOSS paper  
- [ ] Speed up the code
- [ ] GPU programming 
- [ ] dtwFun2 and dtwFun_short are giving slightly different results. 
- [ ] Lighter and faster DTW cost calculation + make the band from long side so it is more accurate. 
- [ ] macOS integration:
  - [ ] Integration for M1/M2 chip machines.  
- [ ] Time shows wrong on macOS with std::clock. Therefore moving to chrono library.
- [x] Clusters class is created to decouple representation of clusters. 
- [x] mip.hpp and mip.cpp files are created to contain MIP functions.
- [ ] Give error message if data could not be loaded. 
- Benchmarking:
  - [x] UCR_test_2018 is continuing. 
  - [ ] USR_dtai.py
  - [ ] TSlearn 
  - [ ] dtwclust in R
- [ ] Encapsulating Data and related functions in one folder. 
- [ ] Open-source solver addition. 
  - [ ] OSQP is added. 
- [ ] Exploration of totally unimodular matrices. 
- [ ] Creating DTW objects taking distance/band as a policy-based design 
- [ ] TestNumberOfThreads should be an atomic counter instead of printing. 
- [ ] Doxygen website? 
- [ ] Remove unnecessary warping functions. 
- [ ] Fix warnings. Especially, we should not get warnings from external libraries. 
- [ ] Make Gurobi dependency optional. (Now it cannot be disabled. )
- [ ] Consider including Eigen library for matrix operations / linear system solution. 
- [ ] w based DTW. 
- [ ] Reading memoisation matrix distMat from file instead of re-calculating DTW every time. 

### Low priority: 
- N.A.

### Open questions:
- [ ] How to create a user interface so that they can use examples as helper functions? There are many parameters. Maybe config file? 

### New ideas:
- [ ] 2-D /N-D DTW
- [ ] DTW for irregular data. 
- [ ] only allow warping at 0 values
- [ ] change no band input to -1 instead of 0 to allow users to have 0 warping (euclidean distance) if desired

### Formatting: 
- [ ] Configure clang-format, cmake-format etc. 

### Developer changes: 
- [ ] CMake folder and some files are added. 

### JOSS: 
- [ ] Added JOSS folder and Github workflow. 