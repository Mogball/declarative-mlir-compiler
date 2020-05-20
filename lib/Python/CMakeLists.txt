Python3_add_library(mlir
  Context.cpp
  Context.h
  OwningModuleRef.cpp
  OwningModuleRef.h
  Parser.cpp
  Parser.h
  Module.cpp
  Module.h

  Expose.cpp
  Expose.h
  ExposeParser.cpp
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
