---
layout: default
title: Command line interface (CLI)
nav_order: 2
---


# Command line interface (CLI)

It is possible to use DTW-C++ from the command line after successfully compiling the code. Please compile the software using [these instructions](1_installation.md) and run the `bin/dtwc_cl` executable. This will provide you with all the command line interface (CLI) functions. To call the CLI from any other folder, you need to add the `/bin` folder into your path. Alternatively, you can just copy the executable into any folder of your choice. 

## Features
- **Multiple clustering methods**: Supports the k-medoids and MIP methods.
- **Customizable iterations**: Users can set the maximum number of iterations for the k-medoids algorithm.
- **Flexible input handling**: Allows users to skip rows and columns in input data.
- **Multiple solver support**: Includes support for [HiGHS](https://highs.dev) and [Gurobi](https://www.gurobi.com) solvers.

## Available options

DTW-C++ provides a command line interface (CLI) for easy interaction. Below are the available command line options:

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


## Example usage

The following instruction will, as an example, read in data from the file `dummy`, search for 5 clusters, skip the first row and column in the datasets, terminate after 5 repetitions, and use the mixed integer programming method.

```bash
dtwc_cl.exe -i "../data/dummy" --Nc=5 --skipRows 1 --skipCols 1 --Nrep=5 --method=mip
```

