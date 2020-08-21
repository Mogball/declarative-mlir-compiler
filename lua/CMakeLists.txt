#set(LUAC_EXE $<TARGET_FILE:luac>)
#set(MLIR_TRANSLATE_EXE $<TARGET_FILE:mlir-translate>)
#set(CLANG_EXE $<TARGET_FILE:clang>)

find_program(ANTLR NAMES antlr antlr4 REQUIRED)
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
