---
layout: default
title: Command line interface (CLI)
nav_order: 6
---


# Command line interface (CLI)

It is possible to use DTWC++ from command line after a successful compilation. Please compile the software using instructions in [running](run.md) page and use `bin/dtwc_cl` executable after compilation. This executable will provide you all the command line interface (CLI) functions. For calling it from any folder you may add `/bin` folder into your path. Alternatively, you may just copy the executable into your folder of choice. 

## Features
- **Multiple Clustering Methods**: Supports kMedoids and MIP methods.
- **Customizable Iterations**: Users can set maximum iterations for iterative algorithms.
- **Flexible Input Handling**: Allows skipping rows and columns in input data.
- **Multiple Solver Support**: Includes support for HiGHS and Gurobi solvers.

## Available options

The application provides a command line interface (CLI) for easy interaction. Below are the available command line options:

```bash
--Nc, --clusters, --number_of_clusters <string>: Set the number of clusters in the format i..j or a single number i.
--name, --probName <string>: Name of the clustering problem.
-i, --in, --input <string>: Path to the input file or folder.
-o, --out, --output <string>: Path to the output folder.
--skipRows <int>: Number of initial rows to skip.
--skipCols, --skipColumns <int>: Number of initial columns to skip.
--maxIter, --iter <int>: Maximum number of iterations.
--method <string>: Clustering method (either kMedoids or MIP).
--repeat, --Nrepeat, --Nrepetition, --Nrep <int>: Number of repetitions for Kmedoids.
--solver, --mip_solver, --mipSolver <string>: Specify the solver to use.
--bandwidth, --bandw, --bandlength <int>: Width of the band used.
```


# Example usage: 

```
dtwc_cl.exe -i "../data/dummy" --Nc=5 --skipRows 1 --skipCols 1 --Nrep=5 --method=mip
```

