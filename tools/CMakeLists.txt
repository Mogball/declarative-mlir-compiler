add_executable(spec spec.cpp)
target_link_libraries(spec
  DMCDynamic
  DMCIO
  DMCSpec
  DMCTraits
  DMCEmbed
  LLVMSupport
  MLIRParser
  DMCEmbedInit
  )

add_executable(gen dialectgen.cpp)
target_link_libraries(gen
  DMCSpec
  DMCDynamic
  DMCTraits
  DMCEmbed
  LLVMSupport
  MLIRStandardOps
  MLIRParser
  MLIRLLVMIR
  MLIRSCFToStandard
  MLIRStandardToLLVM
  DMCEmbedInit
  )
