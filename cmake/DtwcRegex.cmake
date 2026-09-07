include_guard(GLOBAL)

# _dtwc_regex_at_least(<n> <out-var>)
# Regex matching every decimal integer >= n (n >= 1, no leading zeros), in the
# CMake regex dialect, which has no {m,n} repetition. Used to floor Catch2's
# "All tests passed (A assertions in C test cases)" summary without pinning an
# upper bound (adding a test must never fail the gate).
function(_dtwc_regex_at_least n out_var)
  if(NOT n MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "_dtwc_regex_at_least: n must be a positive integer, got '${n}'")
  endif()
  string(LENGTH "${n}" digits)
  # Same digit count: n itself, or n's prefix up to position p, a larger digit
  # at p, then free digits.
  set(alternatives "${n}")
  math(EXPR last "${digits} - 1")
  foreach(p RANGE 0 ${last})
    string(SUBSTRING "${n}" 0 ${p} prefix)
    string(SUBSTRING "${n}" ${p} 1 d)
    if(d LESS 9)
      math(EXPR d1 "${d} + 1")
      math(EXPR free "${digits} - ${p} - 1")
      string(REPEAT "[0-9]" ${free} tail)
      list(APPEND alternatives "${prefix}[${d1}-9]${tail}")
    endif()
  endforeach()
  # More digits than n.
  string(REPEAT "[0-9]" ${digits} same)
  list(APPEND alternatives "[1-9]${same}[0-9]*")
  list(JOIN alternatives "|" joined)
  set(${out_var} "(${joined})" PARENT_SCOPE)
endfunction()
