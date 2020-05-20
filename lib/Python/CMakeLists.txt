pybind11_add_module(mlir
  Context.cpp
  Context.h
  OwningModuleRef.cpp
  OwningModuleRef.h
  Parser.cpp
  Parser.h
  Module.cpp
  Module.h
  Location.cpp
  Location.h
  Identifier.cpp
  Identifier.h
  Type.cpp
  Type.h

  ExposeParser.cpp
  ExposeModule.cpp
  ExposeLocation.cpp
  ExposeType.cpp
  Expose.cpp
  Expose.h
  )

target_include_directories(mlir PUBLIC
  ${Python3_INCLUDE_DIRS}
  )
target_link_libraries(mlir PUBLIC
  ${Python3_LIBRARIES}
  MLIRIR
  MLIRParser
  DMCDynamic
  DMCSpec
  DMCTraits
  )
