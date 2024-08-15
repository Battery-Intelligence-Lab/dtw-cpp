---
layout: default
title: Running
nav_order: 2
---

# Using DTW-C++ 

DTW-C++ does not offer any binaries or wrappers in other languages at the moment. Therefore, the only way to use DTW-C++ is to compile it from C++ source files. With the appropriate compilers and dependencies installed you can easily compile DTW-C++ and use it, for example:

- Edit `main.cpp` in the `dtwc` folder and use the `dtwc_main` executable after compilation using the examples in `examples` folder. 
- Use DTW-C++ from the command line interface, by using the `dtwc_cl` executable after compilation. 
- Use DTW-C++ as an external library in your C++ project by linking the `dtwc++` target in your project. Download the source code to your folder of preference, include the line `add_subdirectory(dtw-cpp)` in your `CMakeLists.txt` file. Then link your library. Alternatively, you may also use [CPM](https://github.com/cpm-cmake/) to interactively download and include DTW-C++. However, it should be noted that including DTWC++ may make the predefined path variables such as `dtwc::settings::dataPath` invalid. Therefore, you may manually define the required paths depending on the structure of your folders. 

# Dependencies

DTW-C++ aims to 





## Building from the source

DTW-C++ aims to be compatible with different compilers and platforms. 


You may easily install DTW-C++ using CMake (although it is not an absolute requirement). Therefore, you need a suitable C++ compiler (preferably [GCC](https://gcc.gnu.org/)) and [CMake](https://cmake.org/) to follow this installation guide.   


### Linux

Generally, both GCC and CMake are installed by default on Linux platforms. However, in some cases you may need to install them. For example, Ubuntu 18.04 comes with an older compiler that does not support some of the functionalities in this code directly. Therefore, you may want to [install a newer version of GCC](https://linuxize.com/post/how-to-install-gcc-compiler-on-ubuntu-18-04/). After this:

1. Download the repository as a [*.zip file](https://github.com/battery-intelligence-lab/dtw-cpp/archive/refs/heads/main.zip), or clone it using following command: 
    ```bash
    git clone https://github.com/battery-intelligence-lab/dtw-cpp.git
    ```
2. After downloading the source files, you need to create a build folder and go into the build folder:
    ```bash
    cd DTWC++  # Go into the main directory of source files.
    mkdir build # Create a build folder if it is not present.
    cd build # Go into the build directory. 
    ```
3. Then, create Makefiles by running:
    ```bash
    cmake -G "Unix Makefiles" .. 
    ```
4. Compile the files:
    ```bash
    cmake --build . -j32 # Assuming that you are still in the build folder. 
    ```

```note
After this, executable will be ready to run at ```../Release/dtwc_main```. By default ```CMAKE_BUILD_TYPE``` is set to ```Release```. If desired, you may also use one of the other options such as ```Debug```, ```Release```, ```RelWithDebInfo```, ```MinSizeRel```. To build using an alternative build type you must explicitly define a ```CMAKE_BUILD_TYPE``` variable. For example, for building with debug mode, use the command:
```cmake --build . -DCMAKE_BUILD_TYPE=Debug```
For further information on using CMake, please refer to the [CMake guide](https://cmake.org/cmake/help/git-stage/index.html).
```

### Windows

On Windows platforms, you probably need to install CMake and a C++ compiler:

1. Install the latest version of the [CMake binary](https://cmake.org/download/#latest).
2. You can then install a compiler of your choice. If you are going to use GCC, we suggest installing [TDM-GCC](https://jmeubank.github.io/tdm-gcc/download/) (preferably the **MinGW-w64 based edition**). Otherwise, you can install the [Visual Studio Community](https://visualstudio.microsoft.com/vs/community/) IDE which comes with its own compiler.
3. You can then follow steps 1--4 as per the [Linux section](#linux), except that in step 3, you should write following command: 
```bash
cmake -G "MinGW Makefiles" ..  # if you use MinGW GCC compiler.
```
Alternatively, you can create a ```*.sln``` file as well as build files via following command if you use Visual Studio Community:
```bash
cmake -G "Visual Studio 16 2019" ..  # if you use Visual Studio's compiler.
```

```note
If you are using Visual Studio Community, you may also open the folder in Visual Studio directly, without using CMake. See [this page](https://docs.microsoft.com/en-us/visualstudio/ide/develop-code-in-visual-studio-without-projects-or-solutions?view=vs-2019) for detailed explanation.
```
