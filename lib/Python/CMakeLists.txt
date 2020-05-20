Python3_add_library(mlir
  Context.cpp
  Parser.cpp
  Expose.cpp
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
