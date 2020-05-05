add_executable(test main.cpp)
target_link_libraries(test
  DMCDynamic
  DMCIO
  DMCSpec
  DMCTraits
  MLIRAnalysis
  MLIRStandardOps
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
