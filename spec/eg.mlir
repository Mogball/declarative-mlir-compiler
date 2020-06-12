func @lua_example() -> () {
  %0 = lua.get_nil !lua.nil
  %1 = lua.convto_value(%0 : !lua.nil) -> !lua.value
  lua.store_var(%1 : !lua.value) -> "a_variable"

  %2 = lua.load_var "a_variable" -> !lua.value
  %3 = lua.convto_bool(%2 : !lua.value) -> !lua.boolean

  %4 = lua.load_var "lhs_var" -> !lua.value
  %5 = lua.load_var "rhs_var" -> !lua.value
  %6 = lua.add(%4, %5) : (!lua.value, !lua.value) -> !lua.value
  lua.store_var(%4 : !lua.value) -> "result_var"
  return
}
