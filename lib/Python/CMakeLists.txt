python_add_module(mlir
  Context.cpp
  Parser.cpp
  Expose.cpp
  )

target_include_directories(mlir PUBLIC
  ${Boost_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  )
target_link_libraries(mlir PUBLIC
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
  MLIRIR
  MLIRParser
  DMCDynamic
  DMCSpec
  DMCTraits
  )
