func @lua_example() -> () {
  %0 = lua.load_var "a" -> !lua.value
  lua.store_var %0 : !lua.value -> "a"
  return
}
