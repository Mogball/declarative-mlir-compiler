Dialect @lua {
  // Concreate lua value
  Type @val
  Alias @value -> !dmc.Isa<@lua::@val> { builder = "lua.val()" }

  // Multiple assign and return value helpers:
  Type @pack
  Alias @value_pack -> !dmc.Isa<@lua::@pack> { builder = "lua.pack()" }

  Op @concat(vals: !dmc.Variadic<!lua.value>,
             tail: !dmc.Variadic<!lua.value_pack>) -> (pack: !lua.value_pack)
    traits [@SizedOperandSegments, @Alloc<"pack">]
    config { fmt = "`(` operands `)` `:` functional-type(operands, results) attr-dict" }
  Op @unpack(pack: !lua.value_pack) -> (vals: !dmc.Variadic<!lua.value>)
    traits [@SameVariadicResultSizes, @NoSideEffects]
    config { fmt = "$pack `:` functional-type($pack, $vals) attr-dict" }

  // Variable handling
  Op @alloc_local() -> (res: !lua.value) { var = #dmc.String }
    traits [@Alloc<"res">]
    config { fmt = "$var attr-dict" }
  Op @get_or_alloc() -> (res: !lua.value) { var = #dmc.String }
    traits [@Alloc<"res">]
    config { fmt = "$var attr-dict" }
  Op @alloc() -> (res: !lua.value) { var = #dmc.String }
    config { fmt = "$var attr-dict" }
  Op @assign(tgt: !lua.value, val: !lua.value) -> (res: !lua.value)
    traits [@WriteTo<"tgt">]
    config { fmt = "$tgt `=` $val attr-dict" }
  Op @copy(tgt: !lua.value, val: !lua.value) -> ()
    traits [@WriteTo<"tgt">]
    config { fmt = "$tgt `=` $val attr-dict" }

  // Function calls
  Op @call(fcn: !lua.value, args: !lua.value_pack) -> (rets: !lua.value_pack)
    traits [@MemoryWrite]
    config { fmt = "$fcn `(` $args `)` attr-dict" }

  Alias @Builtin -> #dmc.AnyOf<
      "math", "io", "table", "print", "string">

  Op @builtin() -> (val: !lua.value) { var = #lua.Builtin }
    traits [@NoSideEffects]
    config { fmt = "$var attr-dict" }

  // Value getters
  Op @nil() -> (res: !lua.value)
    config { fmt = "attr-dict" }
  Op @boolean() -> (res: !lua.value) { value = #dmc.I<1> }
    config { fmt = "$value attr-dict" }
  Op @number() -> (res: !lua.value) { value = #dmc.F<64> }
    config { fmt = "$value attr-dict" }

  Op @table() -> (res: !lua.value)
    config { fmt = "attr-dict" }
  Op @table_get(tbl: !lua.value, key: !lua.value) -> (val: !lua.value)
    config { fmt = "$tbl `[` $key `]` attr-dict" }
  Op @table_set(tbl: !lua.value, key: !lua.value, val: !lua.value) -> ()
    traits [@WriteTo<"tbl">]
    config { fmt = "$tbl `[` $key `]` `=` $val attr-dict" }

  Op @get_string() -> (res: !lua.value) { value = #dmc.String }
    config { fmt = "$value attr-dict" }

  // Value operations
  Alias @BinaryOp -> #dmc.AnyOf<
      "or", "and",
      "<", ">", "<=", ">=", "==", "~=",
      "..",
      "+", "-", "*", "/", "%", "//",
      "&", "|", "~", "<<", ">>",
      "^">
  Op @binary(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    { op = #lua.BinaryOp }
    config { fmt = "$lhs $op $rhs attr-dict" }

  Alias @UnaryOp -> #dmc.AnyOf<
      "not", "#", "-", "~">
  Op @unary(val: !lua.value) -> (res: !lua.value)
    { op = #lua.UnaryOp }
    config { fmt = "$op $val attr-dict" }

  Op @numeric_for(lower: !lua.value, upper: !lua.value, step: !lua.value) -> ()
    { ivar = #dmc.String } (region: Sized<1>)
    traits [@NoSideEffects, @LoopLike<"region", "licmDefinedOutside", "licmCanHoist">]
    config { fmt = "$ivar `in` `[` $lower `,` $upper `]` `by` $step `do` $region attr-dict" }
  Op @generic_for(f: !lua.value, s: !lua.value, var: !lua.value) -> ()
    { params = #dmc.ArrayOf<#dmc.String> } (region: Sized<1>)
    traits [@MemoryWrite, @LoopLike<"region", "licmDefinedOutside", "licmCanHoist">]
    config { fmt = "$params `in` $f `,` $s `,` $var `do` $region attr-dict" }
  Op @function_def() -> (fcn: !lua.value)
    { params = #dmc.ArrayOf<#dmc.String> } (region: Sized<1>)
    traits [@NoSideEffects]
    config { fmt = "$params $region attr-dict" }
  Op @function_def_capture(captures: !dmc.Variadic<!lua.value>) -> (fcn: !lua.value)
    { params = #dmc.ArrayOf<#dmc.String> } (region: Sized<1>)
    traits [@SameVariadicOperandSizes, @NoSideEffects]
    config { fmt = "`(` operands `)` `:` type(operands) $params $region attr-dict" }
  Op @cond_if(cond: !lua.value) -> () (first: Sized<1>, second: Sized<1>)
    traits [@NoSideEffects]
    config { fmt = "$cond `then` $first `else` $second attr-dict" }
  Op @loop_while() -> () (eval: Any, region: Any)
    traits [@NoSideEffects, @LoopLike<"region", "licmDefinedOutside", "licmCanHoist">]
    config { fmt = "$eval `do` $region attr-dict" }
  Op @repeat() -> () (region: Sized<1>)
    traits [@NoSideEffects, @LoopLike<"region", "licmDefinedOutside", "licmCanHoist">]
    config { fmt = "$region attr-dict" }
  Op @until() -> () (eval: Sized<1>)
    traits [@IsTerminator, @LoopLike<"region", "licmDefinedOutside", "licmCanHoist">]
    config { fmt = "$eval attr-dict" }
  Op @end() -> ()
    traits [@IsTerminator]
    config { fmt = "attr-dict" }
  Op @ret(vals: !dmc.Variadic<!lua.value>,
          tail: !dmc.Variadic<!lua.value_pack>) -> ()
    traits [@IsTerminator, @SizedOperandSegments]
    config { fmt = "`(` operands `)` `:` type(operands) attr-dict" }
  Op @cond(cond: !lua.value) -> ()
    traits [@IsTerminator]
    config { fmt = "$cond attr-dict" }

  /// Function capture
  Type @capture
  Alias @capture_pack -> !dmc.Isa<@lua::@capture> { builder = "lua.capture()" }

  Op @make_capture(vals: !dmc.Variadic<!lua.value>) -> (capture: !lua.capture_pack)
    traits [@NoSideEffects, @SameVariadicOperandSizes]
  Op @get_captures(capture: !lua.capture_pack) -> (vals: !dmc.Variadic<!lua.value>)
    traits [@NoSideEffects, @SameVariadicResultSizes]
}

Dialect @luaopt {
  Op @const_number() -> (res: !lua.value) { value = #dmc.F<64> }
    traits [@NoSideEffects]

  Alias @table_prealloc -> 4
  Op @table_get_prealloc(tbl: !lua.value, iv: i64) -> (val: !lua.value)
  Op @table_set_prealloc(tbl: !lua.value, iv: i64, val: !lua.value) -> ()
    traits [@WriteTo<"tbl">]

  Op @capture_self(val: !lua.value) -> (res: !lua.value)
    traits [@NoSideEffects]
}

Dialect @luac {
  Alias @_int -> i64 { builder = "IntegerType(64)" }

  Alias @bool -> i1 { builder = "IntegerType(1)" }
  Alias @real -> f64 { builder = "F64Type()" }
  Alias @pack_fcn -> (!lua.capture, !lua.pack) -> !lua.pack
    { builder = "FunctionType([lua.pack(), lua.pack()], [lua.pack()])" }

  Type @ref
  Alias @value_ref -> !dmc.Isa<@luac::@ref> { builder = "luac.ref()" }

  Alias @type_enum -> i16 { builder = "IntegerType(16)" }
  Alias @type_nil -> 0 : i16
  Alias @type_bool -> 1 : i16
  Alias @type_num -> 2 : i16
  Alias @type_str -> 3 : i16
  Alias @type_tbl -> 4 : i16
  Alias @type_fcn -> 5 : i16
  // userdata, thread unimplemented

  Op @wrap_bool(b: !luac.bool) -> (res: !lua.value)
    config { fmt = "$b attr-dict" }
  Op @wrap_real(num: !luac.real) -> (res: !lua.value)
    config { fmt = "$num attr-dict" }
  Op @make_fcn(addr: !luac.pack_fcn, capture: !lua.capture_pack) -> (fcn: !lua.value)

  /// Binary operations
  Op @add(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    config { fmt = "`(` operands `)` attr-dict" }
  Op @sub(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    config { fmt = "`(` operands `)` attr-dict" }
  Op @mul(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    config { fmt = "`(` operands `)` attr-dict" }
  Op @pow(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    config { fmt = "`(` operands `)` attr-dict" }
  Op @strcat(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    config { fmt = "`(` operands `)` attr-dict" }
  Op @eq(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    config { fmt = "`(` operands `)` attr-dict" }
  Op @ne(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    config { fmt = "`(` operands `)` attr-dict" }
  Op @lt(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    config { fmt = "`(` operands `)` attr-dict" }
  Op @le(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    config { fmt = "`(` operands `)` attr-dict" }
  Op @gt(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    config { fmt = "`(` operands `)` attr-dict" }
  Op @bool_and(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    config { fmt = "`(` operands `)` attr-dict" }

  /// Unary operations
  Op @bool_not(val: !lua.value) -> (res: !lua.value)
    config { fmt = "$val attr-dict" }
  Op @list_size(val: !lua.value) -> (res: !lua.value)
    config { fmt = "$val attr-dict" }
  Op @neg(val: !lua.value) -> (res: !lua.value)
    config { fmt = "$val attr-dict" }

  /// Misc library functions
  Op @convert_bool_like(val: !lua.value) -> (b: !luac.bool)
    config { fmt = "$val attr-dict" }

  /// TObject -> TObject *
  Op @get_ref(val: !lua.value) -> (ref: !luac.value_ref)
    config { fmt = "$val attr-dict" }

  Op @alloc() -> (res: !lua.value)
    config { fmt = "attr-dict" }
  Op @alloc_gc(val: !luac.value_ref) -> ()
    traits [@WriteTo<"val">]
    config { fmt = "$val attr-dict" }

  Op @get_type(val: !lua.value) -> (ty: !luac.type_enum)
    config { fmt = "`type` `(` $val `)` attr-dict" }
  Op @set_type(ptr: !luac.value_ref, ty: !luac.type_enum) -> ()
    traits [@WriteTo<"ptr">]
    config { fmt = "`type` `(` $ptr `)` `=` $ty attr-dict" }

  Op @get_bool_val(val: !lua.value) -> (b: !luac.bool)
    config { fmt = "$val attr-dict" }
  Op @set_bool_val(ptr: !luac.value_ref, b: !luac.bool) -> ()
    traits [@WriteTo<"ptr">]
    config { fmt = "$ptr `=` $b attr-dict" }

  Op @get_double_val(val: !lua.value) -> (num: !luac.real)
    config { fmt = "$val attr-dict" }
  Op @set_double_val(ptr: !luac.value_ref, num: !luac.real) -> ()
    traits [@WriteTo<"ptr">]
    config { fmt = "$ptr `=` $num attr-dict" }

  Op @get_fcn_addr(val: !lua.value) -> (addr: !luac.pack_fcn)
    config { fmt = "$val attr-dict" }
  Op @set_fcn_addr(ptr: !luac.value_ref, addr: !luac.pack_fcn) -> ()
    traits [@WriteTo<"ptr">]
    config { fmt = "$ptr `=` $addr attr-dict" }

  Op @get_capture_pack(val: !lua.value) -> (capture: !lua.capture_pack)
    config { fmt = "$val `[` `]` attr-dict" }
  Op @set_capture_pack(ptr: !luac.value_ref, capture: !lua.capture_pack) -> ()
    traits [@WriteTo<"ptr">]
    config { fmt = "$ptr `[` `]` `=` $capture attr-dict" }

  Op @get_value_union(val: !lua.value) -> (u: i64)
    config { fmt = "$val attr-dict" }
  Op @set_value_union(ptr: !luac.value_ref, u: i64) -> ()
    traits [@WriteTo<"ptr">]
    config { fmt = "$ptr `=` $u attr-dict" }

  Op @new_capture(size: i32) -> (capture: !lua.capture_pack)
  Op @add_capture(capture: !lua.capture_pack, ptr: !luac.value_ref, idx: i32) -> ()
    traits [@WriteTo<"capture">]

  Op @get_ret_pack(size: i32) -> (pack: !lua.value_pack)
  Op @get_arg_pack(size: i32) -> (pack: !lua.value_pack)

  Op @pack_insert(pack: !lua.value_pack, val: !lua.value, idx: i32) -> ()
    traits [@WriteTo<"pack">]
  Op @pack_insert_all(pack: !lua.value_pack, tail: !lua.value_pack, idx: i32) -> ()
    traits [@WriteTo<"pack">]
  Op @pack_get(pack: !lua.value_pack, idx: i32) -> (res: !lua.value)
  Op @pack_get_size(pack: !lua.value_pack) -> (size: i32)

  Op @global_string() -> () { sym = #dmc.String, value = #dmc.String }
    traits [@MemoryWrite] config { fmt = "symbol($sym) `=` $value attr-dict" }
  Op @load_string() -> (res: !lua.value) { global_sym = #dmc.String }
    config { fmt = "symbol($global_sym) attr-dict" }
}

Dialect @luallvm {
  Alias @value -> !llvm<"{ i32, { i64 } }">
  Alias @ref -> !llvm<"{ i32, { i64 } }*">
  Alias @pack -> !llvm<"{ i64, i64, { i32, { i64 } }** }*">

  Op @load_string(data: !llvm<"i8*">, length: !llvm.i64) -> (val: !luallvm.value)
    config { fmt = "`(` operands `)` `:` functional-type(operands, results) attr-dict" }
}
