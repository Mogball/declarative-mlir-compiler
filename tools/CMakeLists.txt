add_executable(test main.cpp)
target_link_libraries(test
  DMCDynamic
  DMCIO
  DMCSpec
  DMCTraits
  MLIRStandardOps
  MLIRParser
  )

add_executable(spec spec.cpp)
target_link_libraries(spec
  DMCDynamic
  DMCIO
  DMCSpec
  DMCTraits
  LLVMSupport
  MLIRParser
  )

add_executable(gen dialectgen.cpp)
target_link_libraries(gen
  DMCSpec
  DMCDynamic
  DMCTraits
  LLVMSupport
  MLIRParser
  )
