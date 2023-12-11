/*
 * Method.hpp
 *
 * Method enum for classification method

 *  Created on: 11 Dec 2023
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

namespace dtwc {

enum class Method {
  Kmedoids, //<! Kmedoids classification
  MIP       //<! Mixed integer programming classification
};

}