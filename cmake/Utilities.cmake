# @file Utilities.cmake
# @brief CMake string and Windows environment-capture utilities.
# @author Volkan Kumtepeli
# @date 04 Dec 2023

# Extract substring after a prefix (e.g. VCVARSALL_ENV_START).
function(find_substring_by_prefix output prefix input)
  string(FIND "${input}" "${prefix}" prefix_index)
  if("${prefix_index}" STREQUAL "-1")
    message(SEND_ERROR "Could not find ${prefix} in ${input}")
  endif()
  string(LENGTH "${prefix}" prefix_length)
  math(EXPR start_index "${prefix_index} + ${prefix_length}")
  string(SUBSTRING "${input}" "${start_index}" "-1" _output)
  set("${output}" "${_output}" PARENT_SCOPE)
endfunction()

# Populate CMake ENV from the output of `cmd /c set`.
function(set_env_from_string env_string)
  # Escape ';' in Windows paths before splitting lines into a CMake list.
  string(REGEX REPLACE ";" "__sep__" env_string_sep_added "${env_string}")
  string(REGEX REPLACE "\r?\n" ";" env_list "${env_string_sep_added}")

  foreach(env_var IN LISTS env_list)
    string(FIND "${env_var}" "=" _eq_idx)
    if(_eq_idx GREATER 0)
      string(SUBSTRING "${env_var}" 0 ${_eq_idx} env_name)
      math(EXPR _val_start "${_eq_idx} + 1")
      string(SUBSTRING "${env_var}" ${_val_start} -1 env_value)
      string(REGEX REPLACE "__sep__" ";" env_value "${env_value}")
      set(ENV{${env_name}} "${env_value}")
      if("${env_name}" STREQUAL "PATH")
        list(APPEND CMAKE_PROGRAM_PATH ${env_value})
        set(CMAKE_PROGRAM_PATH "${CMAKE_PROGRAM_PATH}" PARENT_SCOPE)
      endif()
    endif()
  endforeach()
endfunction()
