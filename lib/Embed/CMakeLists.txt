add_library(DMCEmbed
  Constraints.cpp
  Spec.cpp
  Init.cpp
  OpFormatGen.cpp
  TypeFormatGen.cpp
  PythonGen.cpp
  InMemoryDef.cpp
  ParserPrinter.cpp
  FormatUtils.cpp
  FormatUtils.h
  )

target_link_libraries(DMCEmbed PUBLIC
  MLIRIR
  LLVMMLIRTableGen
  pybind11
  pymlir
  )
