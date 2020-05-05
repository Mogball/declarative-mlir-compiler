add_library(DMCSpec
  SpecDialect.cpp
  SpecTypes.cpp
  SpecTypeImplementation.cpp
  SpecOps.cpp
  SpecAttrs.cpp
  SpecAttrImplementation.cpp
  DialectGen.cpp
  )
target_link_libraries(DMCSpec MLIRIR)
