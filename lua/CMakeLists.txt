add_library(LuaToLuaC LuaToLuaC.cpp)
target_link_libraries(LuaToLuaC
  MLIRIR
  DMCDynamic
  )

add_executable(luac luac.cpp)
target_link_libraries(luac
  LuaToLuaC
  DMCSpec
  DMCTraits
  MLIRParser
  MLIRLLVMIR
  )
