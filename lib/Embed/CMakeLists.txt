add_library(DMCEmbed
  Constraints.cpp
  Spec.cpp
  Init.cpp
  OpFormatGen.cpp
  )

target_link_libraries(DMCEmbed PUBLIC
  MLIRIR
  pybind11
  pymlir
  )
