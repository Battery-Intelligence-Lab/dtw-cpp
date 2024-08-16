/**
 * @file Method.hpp
 * @brief Method enum for classification method.
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 11 Dec 2023
 */

#pragma once

namespace dtwc {

enum class Method {
  Kmedoids, //<! Kmedoids classification
  MIP       //<! Mixed integer programming classification
};

}