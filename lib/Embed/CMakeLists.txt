add_library(DMCEmbed
  Constraints.cpp
  Init.cpp
  )

target_link_libraries(DMCEmbed PUBLIC
  MLIRIR
  pybind11
  pymlir
  )
