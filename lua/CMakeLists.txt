add_library(LuaToLuaC
  LuaToLuaC.cpp
  LuaToLLVM.cpp
  LuaToLuaLib.cpp
  )
target_link_libraries(LuaToLuaC PUBLIC
  MLIRIR
  DMCDynamic
  DMCEmbedInit
  )

add_executable(luac luac.cpp)
target_link_libraries(luac
  LuaToLuaC
  DMCSpec
  DMCDynamic
  DMCTraits
  DMCEmbed
  DMCEmbedInit
  MLIRParser
  MLIRLLVMIR
  MLIRLoopToStandard
  MLIRStandardToLLVM
  )

set(LUAC_EXE $<TARGET_FILE:luac>)
set(MLIR_TRANSLATE_EXE $<TARGET_FILE:mlir-translate>)
set(CLANG_EXE $<TARGET_FILE:clang>)

find_program(ANTLR antlr4 REQUIRED)
add_custom_command(
  OUTPUT parser/Lua.interp
         parser/Lua.tokens
         parser/LuaLexer.interp
         parser/LuaLexer.py
         parser/LuaLexer.tokens
         parser/LuaListener.py
         parser/LuaParser.py
  COMMAND ${ANTLR} -Dlanguage=Python3 -o parser ${CMAKE_CURRENT_SOURCE_DIR}/Lua.g4
  DEPENDS Lua.g4
  COMMENT "Generating Lua parser from ANTLR"
  )
add_custom_target(lua-parser DEPENDS parser/LuaParser.py)

set(luair_file main.mlir)

function(compile_lua_ir OUT IR_FILE DIALECT_FILE LIB_FILE)
  get_filename_component(abs_dialect_file ${DIALECT_FILE} ABSOLUTE)
  get_filename_component(abs_lib_file ${LIB_FILE} ABSOLUTE)
  get_filename_component(abs_ir_file ${IR_FILE} ABSOLUTE)
  get_filename_component(abs_ir_dir ${abs_ir_file} DIRECTORY)
  get_filename_component(ir_basename ${IR_FILE} NAME_WE)
  file(RELATIVE_PATH rel_ir_dir ${CMAKE_CURRENT_SOURCE_DIR} ${abs_ir_dir})
  set(prefix_dir "${CMAKE_CURRENT_BINARY_DIR}/${rel_ir_dir}")
  set(mlir_out "${prefix_dir}/${ir_basename}_llvm.mlir")
  set(llvm_out "${prefix_dir}/${ir_basename}.ll")
  set(obj_out "${prefix_dir}/${ir_basename}.o")
  add_custom_command(
    OUTPUT ${mlir_out}
    COMMAND ${LUAC_EXE} ${abs_ir_file} -dialect=${abs_dialect_file} -lualib=${abs_lib_file} -lower-all -o ${mlir_out}
    DEPENDS ${abs_ir_file} ${abs_dialect_file} ${abs_lib_file}
    COMMENT "Lowering LUA IR ${IR_FILE}"
    )
  add_custom_command(
    OUTPUT ${llvm_out}
    COMMAND ${MLIR_TRANSLATE_EXE} --mlir-to-llvmir ${mlir_out} -o ${llvm_out}
    DEPENDS ${mlir_out}
    COMMENT "Generating LLVM IR ${mlir_out}"
    )
  add_custom_command(
    OUTPUT ${obj_out}
    COMMAND ${CLANG_EXE} -c ${llvm_out} -o ${obj_out} -fpie -Wno-override-module
    DEPENDS ${llvm_out}
    COMMENT "Compiling ${llvm_out}"
    )
  set(${OUT} ${obj_out} PARENT_SCOPE)
endfunction()

set(DIALECT_FILE dialect.mlir)
set(LIB_FILE lib.mlir)
compile_lua_ir(MAIN_FILE main.mlir ${DIALECT_FILE} ${LIB_FILE})

include(CheckPIESupported)
check_pie_supported()
add_executable(main impl.cpp ${MAIN_FILE})
set_property(TARGET main PROPERTY POSITION_INDEPENDENT_CODE TRUE)
set_source_files_properties(
  ${MAIN_FILE}
  PROPERTIES
  EXTERNAL_OBJECT true
  GENERATED true
  )
