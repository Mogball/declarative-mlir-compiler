Python3_add_library(mlir
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

  ExposeParser.cpp
  ExposeModule.cpp
  ExposeLocation.cpp
  Expose.cpp
  Expose.h
  )

target_include_directories(mlir PUBLIC
  ${Boost_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
  )
target_link_libraries(mlir PUBLIC
  ${Boost_LIBRARIES}
  ${Python3_LIBRARIES}
  MLIRIR
  MLIRParser
  DMCDynamic
  DMCSpec
  DMCTraits
  )
