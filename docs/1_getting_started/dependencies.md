---
layout: default
title: Dependencies
nav_order: 3
---

# Dependencies

Here is given the list of dependencies that needs to be installed to use DTWC++ software. 

Dependencies that needs to be installed by the user if they do not exist:
- CMake
- OpenMP
- A suitable compiler (Clang, GCC, MSVC, etc.)

Dependencies that are installed by the CPM package manager: 
- HiGHS 
- CLI11
- Catch2 (for testing)

See `cmake/Dependencies.cmake` for a detailed list of libraries/packages installed by CPM.

## CMake

CMake is a metabuild system required to provide the correct compilation commands to the compiler you may follow the official guidelines to download it for your preference of operating system. 

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