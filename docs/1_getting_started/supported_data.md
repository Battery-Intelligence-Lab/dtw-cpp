---
layout: default
title: Supported data formats
nav_order: 5
---


# Supported data formats

DTW-C++ supports importing data from a limited range of data structures at present. We assume that every time series is one-dimensional. There is no-support at the moment for multidimensional input data. Input data can either be read in from multiple files (one per time series), or from a single file, where each time series is a separate row.

## Reading data from disk

You can specify either a _file path_ or a _folder path_ for your data. The software accommodates `*.csv` and `*.tsv` file extensions.

### Specifying a file path

A _file path_ points to a single file containing all of your data, represented in variable-length rows. The values might be separated by commas, tabs, or spaces. In this scenario, time-series names are allocated sequentially from 1 to N, row by row.

**Example:** A file with 5 time-series of varying lengths:

|       |       |       |       |       |       |       |       |       |
|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| 43.87 | 48.98 | 27.60 | 49.84 | 75.13 | 95.93 |       |       |       |
| 38.16 | 44.56 |       |       |       |       |       |       |       |
| 76.55 | 64.63 | 65.51 | 6.10  | 29.11 | 13.86 | 81.43 | 25.11 |       |
| 79.52 | 70.94 | 16.26 | 58.53 | 69.91 | 14.93 | 24.35 | 61.60 | 12.71 |
| 18.69 | 75.47 | 11.90 | 22.38 |       |       |       |       |       |

### Specifying a folder path

A _folder path_ can contain multiple individual files, each representing a _single_ dataset. Data in these files are read from only a single column. If a file contains multiple columns, it's necessary to skip some to extract just the last column. The values may be separated by commas, tabs, or spaces. In this case, time-series names are derived from the individual file names.

**Example:** A `file.csv` is available, with two columns separated by a comma; the first column is just an index, and the second column is the actual data. We do not need the index values (and cannot process them) so the index column should be skipped. Note that files without headers (such as those generated by Pandas) may display `,0` for the first row. To access the actual data points in such files, specify `skipRows=1` and `skipColumns=1`. Please check your files carefully before running DTW-C++!

|   | , | 0     |
|---|---|-------|
| 0 | , | 43.87 |
| 1 | , | 48.98 |
| 2 | , | 27.60 |
| 3 | , | 49.84 |
| 4 | , | 75.13 |
| 5 | , | 95.93 |

## Reading data directly

If you are using DTW-C++ directly (e.g., as a library within your software), you might prefer to read data independently or use pre-generated data. DTW-C++ employs the `Data` class to encapsulate a `std::vector<std::vector<data_type>>` data object and `std::vector<std::string>` for their corresponding names. The following example code snippet demonstrates how to input data into a Problem object.

```cpp
// Your data generation routine:
std::vector<std::vector<double>> myData = generate_some_data();
std::vector<std::string> myNames = generate_some_names();
//-------------------------------

// Create a Problem object:
dtwc::Problem myProblem{'problem_name'};
auto myDataObject = dtwc::Data(std::move(myData), std::move(myNames));
myProblem.set_data(myDataObject);
/* Other settings */
```
