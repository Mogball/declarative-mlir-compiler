add_library(LuaToLuaC
  LuaToLuaC.cpp
  LuaToLLVM.cpp
  LuaToLuaLib.cpp
  )
target_link_libraries(LuaToLuaC
  MLIRIR
  DMCDynamic
  )

add_executable(mluac luac.cpp)
target_link_libraries(mluac
  LuaToLuaC
  DMCSpec
  DMCTraits
  MLIRParser
  MLIRLLVMIR
  MLIRLoopToStandard
  MLIRStandardToLLVM
  )
