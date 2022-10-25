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

### Low priority: 

- N.A.


### New ideas:
- [ ] 2-D /N-D DTW
- [ ] DTW for irregular data. 


### Formatting: 
- [x] Include a clang-format file. 
- [ ] Configure clang-format, cmake-format etc. 

### Developer changes: 

- [ ] CMake folder and some files are added. 

### JOSS: 
- [ ] Added JOSS folder and Github workflow. 

