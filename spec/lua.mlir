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

  Alias @concrete -> !dmc.AnyOf<!dmc.Isa<@lua::@nil>,
                                !dmc.Isa<@lua::@boolean>,
                                !dmc.Isa<@lua::@string>,
                                !dmc.Isa<@lua::@function>,
                                !dmc.Isa<@lua::@userdata>,
                                !dmc.Isa<@lua::@thread>,
                                !dmc.Isa<@lua::@number>>

  //--------------------------------------------------------------------------//
  // Attributes
  //--------------------------------------------------------------------------//
  Attr @local
  Attr @global
  Alias @var_scope -> #dmc.AnyOf<#dmc.Isa<@lua::@local>, #dmc.Isa<@lua::@global>>
  Alias @scope -> #dmc.Default<#lua.var_scope, #lua.global>

  //--------------------------------------------------------------------------//
  // High-Level Ops
  //--------------------------------------------------------------------------//
  Op @load_var() -> (res: !lua.value)  { var = #dmc.String, scope = #lua.scope }
    config { fmt = "$var `->` type($res) (`at` $scope^)? attr-dict" }
  Op @store_var(val: !lua.value) -> () { var = #dmc.String, scope = #lua.scope }
    config { fmt = "`(` $val `:` type($val) `)` `->` $var (`at` $scope^)? attr-dict" }

  Op @add(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    config { fmt = "`(` operands `)` `:` functional-type(operands, results) attr-dict" }

  //--------------------------------------------------------------------------//
  // Concrete Type Ops
  //--------------------------------------------------------------------------//
  Op @get_nil() -> (res: !lua.nil)
    config { fmt = "type($res) attr-dict" }
  Op @convto_bool(val: !lua.value) -> (res: !lua.boolean)
    config { fmt = "`(` $val `:` type($val) `)` `->` type($res) attr-dict" }
  Op @convto_value(val: !lua.concrete) -> (res: !lua.value)
    config { fmt = "`(` $val `:` type($val) `)` `->` type($res) attr-dict" }
}
