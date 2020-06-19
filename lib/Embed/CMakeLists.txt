add_library(DMCEmbed
  Constraints.cpp
  Spec.cpp
  OpFormatGen.cpp
  TypeFormatGen.cpp
  PythonGen.cpp
  InMemoryDef.cpp
  ParserPrinter.cpp
  Expose.cpp
  FormatUtils.cpp
  FormatUtils.h
  Scope.cpp
  )

target_link_libraries(DMCEmbed PUBLIC
  MLIRIR
  LLVMMLIRTableGen
  pybind11
  pymlir
  )

add_library(DMCEmbedInit Init.cpp)
target_link_libraries(DMCEmbedInit PUBLIC
  pybind11
  pymlir
  )
