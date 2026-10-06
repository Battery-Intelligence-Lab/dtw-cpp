target_sources(dtwc_core
  PRIVATE
    fast_pam.cpp
    fast_clara.cpp
    one_batch_pam.cpp
    barycenter.cpp
    hierarchical.cpp
    tadpole.cpp
  PUBLIC FILE_SET HEADERS FILES
    detail/fast_clara_plan.hpp
    fast_pam.hpp
    fast_clara.hpp
    one_batch_pam.hpp
    barycenter.hpp
    hierarchical.hpp
    tadpole.hpp
  )
