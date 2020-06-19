add_library(pymlir
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
  Attribute.cpp
  Attribute.h

  BuildableType.cpp
  DialectAsm.cpp
  OpAsm.cpp
  Utility.h
  AsmUtils.h

  # Reduce compilation time by splitting the functions
  ExposeBuilder.cpp
  ExposeDialectAsm.cpp
  ExposeOpAsm.cpp
  ExposeOps.cpp
  ExposeValue.cpp
  ExposeShapedTypes.cpp
  ExposeStandardTypes.cpp
  ExposeOpaqueType.cpp
  ExposeFunctionType.cpp
  ExposeElementsAttr.cpp
  ExposeSymbolRefAttr.cpp
  ExposeIntFPAttr.cpp
  ExposeDictAttr.cpp
  ExposeArrayAttr.cpp
  ExposeAttribute.cpp
  ExposeParser.cpp
  ExposeModule.cpp
  ExposeLocation.cpp
  ExposeType.cpp
  Expose.cpp
  Expose.h
  )

target_include_directories(pymlir PUBLIC
  ${Python3_INCLUDE_DIRS}
  )
target_link_libraries(pymlir PUBLIC
  ${Python3_LIBRARIES}
  pybind11
  MLIRIR
  MLIRParser
  MLIRStandardOps
  MLIRLLVMIR
  MLIRSCF
  )

add_library(DMCDLLInit DllInit.cpp)
target_link_libraries(DMCDLLInit PUBLIC
  pybind11
  pymlir
  )

pybind11_add_module(mlir ExternalModule.cpp)
target_link_libraries(mlir PUBLIC
  pymlir
  DMCDynamic
  DMCSpec
  DMCTraits
  DMCDLLInit
  )
