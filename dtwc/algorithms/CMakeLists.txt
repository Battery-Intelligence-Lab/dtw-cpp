target_sources(dtwc_core
  PRIVATE
    fast_pam.cpp
    fast_clara.cpp
    one_batch_pam.cpp
    barycenter.cpp
    hierarchical.cpp
    tadpole.cpp
  PUBLIC FILE_SET HEADERS FILES
    detail/fast_clara_assign.hpp
    detail/fast_clara_plan.hpp
    fast_pam.hpp
    fast_clara.hpp
    one_batch_pam.hpp
    barycenter.hpp
    hierarchical.hpp
    tadpole.hpp
  )
# FastCLARA streaming its series from Parquet needs the Parquet reader: dtwc_io's.
if(TARGET dtwc_io)
  target_sources(dtwc_io PRIVATE fast_clara_parquet.cpp)
endif()
