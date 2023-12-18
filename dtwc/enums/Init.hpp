/*
 * Init.hpp
 *
 * Init enum for initialisation type

 *  Created on: 18 Dec 2023
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

namespace dtwc {

enum class Init {
  random, //<! Random initialisaton
  kpp,    //<! K++ initialisation
  custom  //<! Custom initialisation
};
}