---
layout: default
title: Dependencies
nav_order: 2
---

# Dependencies

Here is given the list of dependencies that needs to be installed to use DTWC++ software. 

## CMake

CMake is 






### Linux

Generally, both gcc and CMake are installed by default on Linux platforms. However, in some cases you may need to install them. For example, Ubuntu 18.04 comes with an older compiler that does not support some of the functionalities in this code directly. Therefore, you may want to install a newer version of GCC as shown [here](https://linuxize.com/post/how-to-install-gcc-compiler-on-ubuntu-18-04/). 

1. Then you may download the repository as [*.zip file](https://github.com/battery-intelligence-lab/dtw-cpp/archive/refs/heads/main.zip) or clone it using following command: 
```bash
git clone https://github.com/battery-intelligence-lab/dtw-cpp.git
```
2. After downloading source files, you need to create a build folder and go into the build folder.
```bash
cd DTWC++  # Go into the main directory of source files.
mkdir build # Create a build folder if it is not present.
cd build # Go into the build directory. 
```
3. Then you may create Makefiles by 
```bash
cmake -G "Unix Makefiles" .. 
```
4. Compile the files:
```bash
cmake --build . -j16 # Assuming that you are still in the build folder. 
```

```note
Then the executable will be ready to run at ```../Release/dtwc_main```. By default ```CMAKE_BUILD_TYPE``` is set to ```Release```. If you want, you may also use one of the other options as ```Debug```, ```Release```, ```RelWithDebInfo```, ```MinSizeRel```.To build using alternative build type you may explicitly define ```CMAKE_BUILD_TYPE``` variable. For example, for building with debug mode you may use the following command. For further information on using CMake, please refer to [CMake guide](https://cmake.org/cmake/help/git-stage/index.html)  


```cmake --build . -DCMAKE_BUILD_TYPE=Debug```
```


### Windows

On Windows platforms, you probably need to install CMake and a C++ compiler. 

1. Install the latest version of [CMake binary](https://cmake.org/download/#latest).
2. You may then install a compiler of your choice. However, if you are going to use GCC, we suggest installing [TDM-GCC](https://jmeubank.github.io/tdm-gcc/download/) (preferably **MinGW-w64 based edition**). Otherwise, you may also install [Visual Studio Community](https://visualstudio.microsoft.com/vs/community/) IDE which comes with its own compiler.
3. You may steps 1--4 on [Linux section](#linux) except in step 3, you should write following command: 
```bash
cmake -G "MinGW Makefiles" ..  # if you use MinGW GCC compiler.
```
or you can create ```*.sln``` file as well as build files via following command if you use Visual Studio Community. 
```bash
cmake -G "Visual Studio 16 2019" ..  # if you use Visual Studio's compiler.
```

```note
    If you are using Visual Studio Community, you may also open the folder in Visual Studio directly, without using CMake. 
    See [here](https://docs.microsoft.com/en-us/visualstudio/ide/develop-code-in-visual-studio-without-projects-or-solutions?view=vs-2019) for detailed explanation.
```


## Gurobi Installation: 

### Linux: 

1. Download the installation file. Then extract it to a folder (preferably opt folder) using the following command:

```bash
tar xvfz gurobi9.5.2_linux64.tar.gz  -C /opt/
```

2. Then add necessary variables to the end of your `~/.bashrc` file. 

```bash
export GUROBI_HOME=/opt/gurobi952/linux64
export PATH=$GUROBI_HOME/bin:$PATH
export LD_LIBRARY_PATH=$GUROBI_HOME/lib:$LD_LIBRARY_PATH
```

3. Don't forget to source your `~/.bashrc` file.

```bash
source ~/.bashrc
```

4. Then you need to add a license. Please obtain an academic license from the Gurobi website. Then use `grbgetkey` command to validate your license. 

```bash
grbgetkey HERE-SHOULD-BE-YOUR-LICENSE-KEY
```

5. To test if the installation went correctly, use command line: 

```bash
gurobi_cl $GUROBI_HOME/examples/data/afiro.mps
```

For a visual guide see the video: https://www.youtube.com/watch?v=yNmeG6Wom1o


### macOS:

For a visual guide see the video: https://www.youtube.com/watch?v=ZcL-NmckTxQ