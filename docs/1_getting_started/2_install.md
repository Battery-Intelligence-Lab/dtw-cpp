---
layout: default
title: Installation
nav_order: 2
---

# Installation

DTWpp is yet to offer any binaries or wrappers in any other languages; therefore, the only way to install DTWpp is to compile it from C++ source files. 

## Building from the source


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

