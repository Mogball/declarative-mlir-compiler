func @lua_example() -> () {
  %0 = lua.get_nil !lua.nil
  %1 = lua.convto_value(%0 : !lua.nil)
  lua.store_var %1 -> "a_variable"

  %2 = lua.load_var "a_variable"
  %3 = lua.convto_bool %2 -> !lua.boolean

  %4 = lua.load_var "lhs_var"
  %5 = lua.load_var "rhs_var"
  %6 = lua.add(%4, %5)
  lua.store_var %4 -> "result_var"
  return
}
