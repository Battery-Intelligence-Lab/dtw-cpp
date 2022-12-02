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
  - [x] Old macOS integration is now working with GLOB to find paths for Gurobi. 
  - [x] TBB cannot be used so we are back to thrad-based parallelisation. 
  - [ ] Integration for M1/M2 chip machines.  
- [ ] Time shows wrong on macOS with std::clock. Therefore moving to chrono library.
- [x] Clusters class is created to decouple representation of clusters. 
- [x] mip.hpp and mip.cpp files are created to contain MIP functions.

### Low priority: 

- N.A.


### New ideas:
- [ ] 2-D /N-D DTW
- [ ] DTW for irregular data. 
- [ ] only allow warping at 0 values
- [ ] change no band input to -1 instead of 0 to allow users to have 0 warping (euclidean distance) if desired

### Formatting: 
- [x] Include a clang-format file. 
- [ ] Configure clang-format, cmake-format etc. 

### Developer changes: 

- [ ] CMake folder and some files are added. 

### JOSS: 
- [ ] Added JOSS folder and Github workflow. 

