add_library(DMCTraits
  SpecTraits.cpp
  Registry.cpp
  )
target_link_libraries(DMCTraits
  DMCDynamic
  MLIRIR
  )
