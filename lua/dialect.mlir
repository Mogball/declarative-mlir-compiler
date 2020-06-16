Dialect @lua {
  //--------------------------------------------------------------------------//
  // Types
  //--------------------------------------------------------------------------//
  Type @value
  Alias @Value -> !dmc.Isa<@lua::@value> // TODO implicitly buildable
    { builder = "build_dynamic_type(\"lua\", \"value\")" }

  /// Concrete built-in types.
  Alias @bool -> i1
    { builder = "IntegerType(1)" }

  // Lua uses 64-bit integers and floats
  Alias @real -> f64 { builder = "F64()" }
  Alias @integer -> i64 { builder = "IntegerType(64)" }
  Alias @number -> !dmc.AnyOf<!lua.real, !lua.integer>

  Alias @concrete -> !dmc.AnyOf<!dmc.Isa<@lua::@bool>, !lua.number>

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

  Op @eq(lhs: !lua.Value, rhs: !lua.Value) -> (res: !lua.bool)
    config { fmt = "`(` operands `)` attr-dict" }
  Op @neq(lhs: !lua.Value, rhs: !lua.Value) -> (res: !lua.bool)
    config { fmt = "`(` operands `)` attr-dict" }

  Op @arithmetic(lhs: !lua.Value, rhs: !lua.Value) -> (res: !lua.Value)
    { op = #lua.ArithmeticOp }
    config { fmt = "$op `(` operands `)` attr-dict" }
  Op @relational(lhs: !lua.Value, rhs: !lua.Value) -> (res: !lua.bool)
    { op = #lua.RelationalOp }
    config { fmt = "$op `(` operands `)` attr-dict" }
  Op @bitwise(lhs: !lua.Value, rhs: !lua.Value) -> (res: !lua.integer)
    { op = #lua.BitwiseOp }
    config { fmt = "$op `(` operands `)` attr-dict" }

  //--------------------------------------------------------------------------//
  // Concrete Type Ops
  //--------------------------------------------------------------------------//
  Op @get_nil() -> (res: !lua.Value)
    config { fmt = "attr-dict" }
  Op @new_table() -> (res: !lua.Value)
    config { fmt = "attr-dict" }
  Op @get_string() -> (res: !lua.Value) { value = #dmc.String }
    config { fmt = "$value attr-dict" }

  Op @wrap(val: !lua.concrete) -> (res: !lua.Value)
    config { fmt = "$val `:` type($val) attr-dict" }
  Op @unwrap(val: !lua.Value) -> (res: !lua.concrete)
    config { fmt = "$val `->` type($res) attr-dict" }

  //--------------------------------------------------------------------------//
  // Type querying
  //--------------------------------------------------------------------------//
  Op @typeof(val: !lua.Value) -> (res: !lua.Value)
    config { fmt = "$val attr-dict" }

  //--------------------------------------------------------------------------//
  // Table Ops
  //--------------------------------------------------------------------------//
  Op @table_get(tbl: !lua.Value, key: !lua.Value) -> (res: !lua.Value)
    config { fmt = "$tbl `[` $key `]` attr-dict" }
  Op @table_set(tbl: !lua.Value, key: !lua.Value, value: !lua.Value) -> ()
    config { fmt = "$tbl `[` $key `]` `=` $value attr-dict" }
  Op @table_size(tbl: !lua.Value) -> (res: !lua.Value)
    config { fmt = "$tbl attr-dict" }
}

Dialect @luac {
  Op @wrap_int(val: !lua.integer) -> (res: !lua.Value) config { fmt = "$val attr-dict" }
  Op @wrap_real(val: !lua.real) -> (res: !lua.Value) config { fmt = "$val attr-dict" }
  Op @wrap_bool(val: !lua.bool) -> (res: !lua.Value) config { fmt = "$val attr-dict" }
  Op @unwrap_int(val: !lua.Value) -> (res: !lua.integer) config { fmt = "$val attr-dict" }
  Op @unwrap_real(val: !lua.Value) -> (res: !lua.real) config { fmt = "$val attr-dict" }
  Op @unwrap_bool(val: !lua.Value) -> (res: !lua.bool) config { fmt = "$val attr-dict" }

  Type @string
  Op @global_string() -> () { sym_name = #dmc.String, str = #dmc.String }
    config { fmt = "symbol($sym_name) `(` $str `)` attr-dict" }
  Op @load_string() -> (str: !luac.string, len: i32) { sym = #dmc.String }
    config { fmt = "symbol($sym) `->` `(` type(results) `)` attr-dict" }
  Op @get_string(str: !luac.string, len: i32) -> (res: !lua.Value)
    config { fmt = "`(` operands `)` `:` functional-type(operands, results) attr-dict" }
}

Dialect @luallvm {
  Alias @value -> !llvm<"{ i32, { { { i64 }, i32 } } }*">
    { builder = "get_aliased_type(\"luallvm\", \"value\")" }
}
