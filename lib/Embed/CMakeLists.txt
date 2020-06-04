add_library(DMCEmbed
  Constraints.cpp
  Spec.cpp
  Init.cpp
  OpFormatGen.cpp
  PythonGen.cpp
  InMemoryDef.cpp
  ParserPrinter.cpp
  )

target_link_libraries(DMCEmbed PUBLIC
  MLIRIR
  LLVMMLIRTableGen
  pybind11
  pymlir
  )
