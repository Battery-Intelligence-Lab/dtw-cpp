# The header is listed in every build, as every folder's are; the source needs the switch.
target_sources(dtwc_core PUBLIC FILE_SET HEADERS FILES metal_dtw.hpp)
if(DTWC_ENABLE_METAL AND DTWC_HAS_METAL_DETECTED)
  target_sources(dtwc_core
    PRIVATE
      metal_dtw.mm
    )
  # ARC off: metal_dtw.mm does its own retain/release through __bridge casts.
  set_source_files_properties(metal_dtw.mm TARGET_DIRECTORY dtwc_core PROPERTIES COMPILE_FLAGS "-fno-objc-arc")
  target_compile_definitions(dtwc_core PUBLIC DTWC_HAS_METAL)
  target_link_libraries(dtwc_core PUBLIC
    "-framework Foundation"
    "-framework Metal")
  message(STATUS "Metal enabled for Apple GPU DTW acceleration")
endif()
