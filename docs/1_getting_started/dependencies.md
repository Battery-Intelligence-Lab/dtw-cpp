---
layout: default
title: Dependencies
nav_order: 3
---

# Dependencies

There are several pre-requisite installations required to compile and run DTW-C++.

The following dependencies need to be installed by the user if they do not already exist:
- CMake
- OpenMP
- A suitable compiler (Clang, GCC, MSVC, etc.)
- Gurobi (optional, if not installed then HiGHS will be used as the MIP solver)

The following dependencies are installed by the CPM package manager: 
- HiGHS 
- CLI11
- Catch2 (for testing)

See `cmake/Dependencies.cmake` for a detailed list of libraries/packages installed by CPM.

## CMake

CMake is a metabuild system required to provide the correct compilation commands to the compiler. Please follow the official guidelines to download it for your preference of operating system. 

## Gurobi

Gurobi is a powerful optimisation solver that is free for academic use. If you do not wish to use Gurobi, HiGHS will be used instead.

### Linux installation

1. Download the installation file. Then extract it to a folder (preferably the `opt` folder) using the following command:

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

4. Then you need to add a licence. If you are an academic, obtain an academic licence from the Gurobi website. Then use `grbgetkey` command to validate your licence. 

```bash
grbgetkey HERE-SHOULD-BE-YOUR-LICENSE-KEY
```

5. To test if the installation was correct, use the command: 

```bash
gurobi_cl $GUROBI_HOME/examples/data/afiro.mps
```

There is a [visual guide](https://www.youtube.com/watch?v=yNmeG6Wom1o) available for Linux installation.


### macOS installation

There is a [visual guide](https://www.youtube.com/watch?v=ZcL-NmckTxQ) available for macOS installation.
