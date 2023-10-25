/*
 * SimplexSolver.hpp
 *
 * Sparse implementation of a Simplex table. 

 *  Created on: 22 Oct 2023
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */


#pragma once

#include "../settings.hpp"
#include "../utility.hpp"
#include "solver_util.hpp"


#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>

#include <tuple>
#include <stdexcept>
#include <set>
#include <limits>
#include <map>


namespace dtwc::solver {

class SimplexTable
{
    std::vector<int> basicVariables; 
    std::vector<std::map<int,double>> innerTable;   
    std::vector<double> reducedCosts; 
    std::vector<double> constraintValues;
    double negativeObjective{};  


};



}
