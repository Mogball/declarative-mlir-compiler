add_executable(test main.cpp)
target_link_libraries(test
  DMCDynamic
  DMCIO
  DMCSpec
  MLIRAnalysis
  MLIRStandardOps
  )

add_executable(spec spec.cpp)
target_link_libraries(spec
  DMCDynamic
  DMCIO
  DMCSpec
  LLVMSupport
  MLIRParser
  )
