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
  Op @table_size(tbl: !lua.Value) -> (res: !lua.integer)
    config { fmt = "$tbl attr-dict" }
}

Dialect @luac {
  Op @allocate_object() -> (obj: !lua.Value) config { fmt = "attr-dict" }

  Type @object
  Alias @Object -> !dmc.Isa<@luac::@object>
    { builder = "build_dynamic_type(\"luac\", \"table\")" }

  Op @allocate_table() -> (tbl: !luac.Object) config { fmt = "attr-dict" }
  Op @give_table(obj: !lua.Value, tbl: !luac.Object) -> ()
    config { fmt = "$obj `=` $tbl attr-dict" }

  Op @allocate_string() -> (str: !luac.Object) config { fmt = "attr-dict" }
  Op @give_string(obj: !lua.Value, str: !luac.Object) -> ()
    config { fmt = "$obj `=` $str attr-dict" }
  Op @set_string_bytes(str: !luac.Object) -> () { bytes = #dmc.String }
    config { fmt = "$str `=` $bytes attr-dict" }

  Op @set_integer(obj: !lua.Value, int: !lua.integer) -> ()
    config { fmt = "$obj `=` $int attr-dict" }
  Op @set_real(obj: !lua.Value, real: !lua.real) -> ()
    config { fmt = "$obj `=` $real attr-dict" }
  Op @set_bool(obj: !lua.Value, bool: !lua.bool) -> ()
    config { fmt = "$obj `=` $bool attr-dict" }
}

Dialect @luallvm {
  Alias @value -> !llvm<"{ i32, { { { i64 }, i32 } } }*">
    { builder = "get_aliased_type(\"luallvm\", \"value\")" }
}
