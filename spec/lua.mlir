Dialect @lua {
  //--------------------------------------------------------------------------//
  // Types
  //--------------------------------------------------------------------------//
  Type @value
  Alias @Value -> !dmc.Isa<@lua::@value> // TODO implicitly buildable
    { builder = "build_dynamic_type(\"lua\", \"value\")" }

  /// Concrete built-in types.
  Type @nil
  Alias @Bool -> i1
    { builder = "IntegerType(1)" }
  Type @string
  Alias @String -> !dmc.Isa<@lua::@string>
    { builder = "build_dynamic_type(\"lua\", \"string\")" }
  Type @table
  Alias @Table -> !dmc.Isa<@lua::@table>
    { builder = "build_dynamic_type(\"lua\", \"table\")" }
  Type @function
  Type @userdata
  Type @thread

  // Lua uses 64-bit integers and floats
  Alias @real -> f64 { builder = "F64()" }
  Alias @integer -> i64 { builder = "IntegerType(64)" }
  Alias @number -> !dmc.AnyOf<!lua.real, !lua.integer>

  Alias @concrete -> !dmc.AnyOf<!dmc.Isa<@lua::@nil>,
                                !dmc.Isa<@lua::@bool>,
                                !dmc.Isa<@lua::@string>,
                                !dmc.Isa<@lua::@table>,
                                !dmc.Isa<@lua::@function>,
                                !dmc.Isa<@lua::@userdata>,
                                !dmc.Isa<@lua::@thread>,
                                !lua.number>

  //--------------------------------------------------------------------------//
  // Attributes
  //--------------------------------------------------------------------------//
  Attr @local
  Attr @global
  Alias @var_scope -> #dmc.AnyOf<#dmc.Isa<@lua::@local>, #dmc.Isa<@lua::@global>>
  Alias @scope -> #dmc.Default<#lua.var_scope, #lua.global>

  Alias @None -> !dmc.None { builder = "NoneType()" }
  Alias @ArithmeticOp -> #dmc.AnyOf<"unm", "add", "sub", "mul", "div", "idiv",
                                    "mod", "pow", "concat">
  Alias @BitwiseOp -> #dmc.AnyOf<"and", "or", "xor", "not", "shl", "shr">
  Alias @RelationalOp -> #dmc.AnyOf<"eq", "lt", "le">

  //--------------------------------------------------------------------------//
  // High-Level Ops
  //--------------------------------------------------------------------------//
  Op @add(lhs: !lua.Value, rhs: !lua.Value) -> (res: !lua.Value)
    config { fmt = "`(` operands `)` attr-dict" }
  Op @sub(lhs: !lua.Value, rhs: !lua.Value) -> (res: !lua.Value)
    config { fmt = "`(` operands `)` attr-dict" }

  Op @eq(lhs: !lua.Value, rhs: !lua.Value) -> (res: !lua.Bool)
    config { fmt = "`(` operands `)` attr-dict" }
  Op @neq(lhs: !lua.Value, rhs: !lua.Value) -> (res: !lua.Bool)
    config { fmt = "`(` operands `)` attr-dict" }

  Op @arithmetic(lhs: !lua.Value, rhs: !lua.Value) -> (res: !lua.Value)
    { op = #lua.ArithmeticOp }
    config { fmt = "$op `(` operands `)` attr-dict" }
  Op @relational(lhs: !lua.Value, rhs: !lua.Value) -> (res: !lua.Bool)
    { op = #lua.RelationalOp }
    config { fmt = "$op `(` operands `)` attr-dict" }
  Op @bitwise(lhs: !lua.Value, rhs: !lua.Value) -> (res: !lua.integer)
    { op = #lua.BitwiseOp }
    config { fmt = "$op `(` operands `)` attr-dict" }

  //--------------------------------------------------------------------------//
  // Concrete Type Ops
  //--------------------------------------------------------------------------//
  Op @get_nil() -> (res: !lua.nil)
    config { fmt = "type($res) attr-dict" }

  Op @convto(val: !lua.Value) -> (res: !lua.concrete)
    config { fmt = "$val `->` type($res) attr-dict" }
  Op @tovalue(val: !lua.concrete) -> (res: !lua.Value)
    config { fmt = "`(` $val `:` type($val) `)` attr-dict" }

  Op @typeof(val: !lua.Value) -> (res: !lua.String)
    config { fmt = "`(` $val `)` attr-dict" }

  Op @string_const() -> (res: !lua.String) { value = #dmc.String }
    config { fmt = "$value attr-dict" }

  //--------------------------------------------------------------------------//
  // Table Ops
  //--------------------------------------------------------------------------//
  Op @get(tbl: !lua.Table, key: !lua.Value) -> (res: !lua.Value)
    config { fmt = "$tbl `[` $key `]` attr-dict" }
  Op @size(tbl: !lua.Table) -> (res: !lua.integer)
    config { fmt = "`(` $tbl `)` attr-dict" }
}
