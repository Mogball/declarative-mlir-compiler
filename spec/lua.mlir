Dialect @lua {
  //--------------------------------------------------------------------------//
  // Types
  //--------------------------------------------------------------------------//
  Type @value

  /// Concrete built-in types.
  Type @nil
  Alias @boolean -> i1
  Type @string
  Type @function
  Type @userdata
  Type @thread

  // Lua uses 64-bit integers and floats
  Alias @real -> f64
  Alias @integer -> i64
  Alias @number -> !dmc.AnyOf<!dmc.Isa<@lua::@real>, !dmc.Isa<@lua::@integer>>

  //--------------------------------------------------------------------------//
  // Attributes
  //--------------------------------------------------------------------------//
  Attr @local
  Attr @global
  Alias @Scope -> #dmc.AnyOf<#dmc.Isa<@lua::@local>, #dmc.Isa<@lua::@global>>

  //--------------------------------------------------------------------------//
  // High-Level Ops
  //--------------------------------------------------------------------------//
  Op @load_var() -> (res: !lua.value)  { var = #dmc.String, scope = #lua.Scope }
  Op @store_var(val: !lua.value) -> () { var = #dmc.String, scope = #lua.Scope }
}
