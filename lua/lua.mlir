Dialect @lua {
  // Base Lua type
  Type @ref
  Alias @value -> !dmc.Isa<@lua::@ref> { builder = "lua.ref()" }

  // Multiple assign and return value helpers:
  // %p0 = concat(%2, %3) -> [%2, %3]
  // %p1 = concat(%5, %6) -> [%5, %6]
  // %P = concat(%0, %1, %p0, %4, %p1) -> [%0, %1, ..., %6]
  // %v:3 = unpack(%P) -> %0, %1, %2
  // %v:4 = unpack(%p1) -> %5, %6, nil, nil
  Type @pack
  Alias @value_pack -> !dmc.Isa<@lua::@pack> { builder = "lua.pack()" }
  Alias @value_or_pack -> !dmc.AnyOf<!lua.value_pack, !lua.value>

  Op @concat(vals: !dmc.Variadic<!lua.value_or_pack>) -> (pack: !lua.value_pack)
    traits [@SameVariadicOperandSizes, @ReadFrom<"vals">]
    config { fmt = "`(` $vals `)` `:` functional-type($vals, $pack) attr-dict" }
  Op @unpack(pack: !lua.value_pack) -> (vals: !dmc.Variadic<!lua.value>)
    traits [@SameVariadicResultSizes, @ReadFrom<"pack">, @WriteTo<"pack">]
    config { fmt = "$pack `:` functional-type($pack, $vals) attr-dict" }
  Op @unpack_rewind(pack: !lua.value_pack) -> (vals: !dmc.Variadic<!lua.value>)
    traits [@SameVariadicResultSizes, @ReadFrom<"pack">, @WriteTo<"pack">]
    config { fmt = "$pack `:` functional-type($pack, $vals) attr-dict" }

  // Variable handling
  Op @alloc_local() -> (res: !lua.value) { var = #dmc.String }
    traits [@Alloc<"res">] config { fmt = "$var attr-dict" }
  Op @get_or_alloc() -> (res: !lua.value) { var = #dmc.String }
    traits [@Alloc<"res">] config { fmt = "$var attr-dict" }
  Op @alloc() -> (res: !lua.value) { var = #dmc.String }
    traits [@Alloc<"res">] config { fmt = "$var attr-dict" }
  Op @assign(tgt: !lua.value, val: !lua.value) -> (res: !lua.value)
    { var = #dmc.AnyOf<#dmc.String, #dmc.Unit> }
    traits [@WriteTo<"tgt">, @ReadFrom<"val">]
    config { fmt = "$var $tgt `=` $val attr-dict" }

  // Function calls
  Op @call(fcn: !lua.value, args: !lua.value_pack) -> (rets: !lua.value_pack)
    traits [@MemoryAlloc, @MemoryFree, @MemoryRead, @MemoryWrite]
    config { fmt = "$fcn `(` $args `)` attr-dict" }

  Alias @Builtin -> #dmc.AnyOf<
      "print", "string">

  Op @builtin() -> (val: !lua.value) { var = #lua.Builtin }
    config { fmt = "$var attr-dict" }

  // Value getters
  Op @nil() -> (res: !lua.value)
    traits [@Alloc<"res">] config { fmt = "attr-dict" }
  Op @number() -> (res: !lua.value) { value = #dmc.AnyOf<#dmc.AnyI<64>, #dmc.F<64>> }
    traits [@Alloc<"res">]
    config { fmt = "$value attr-dict" }
  Op @init_table(tbl: !lua.value) -> ()
    traits [@WriteTo<"tbl">] config { fmt = "$tbl attr-dict" }
  Op @table_get(tbl: !lua.value, key: !lua.value) -> (val: !lua.value)
    traits [@WriteTo<"tbl">, @ReadFrom<"key">, @Alloc<"val">]
    config { fmt = "$tbl `[` $key `]` attr-dict" }
  Op @table_set(tbl: !lua.value, key: !lua.value, val: !lua.value) -> ()
    traits [@WriteTo<"tbl">, @ReadFrom<["key", "val"]>]
    config { fmt = "$tbl `[` $key `]` `=` $val attr-dict" }
  Op @get_string() -> (res: !lua.value) { value = #dmc.String }
    traits [@Alloc<"res">] config { fmt = "$value attr-dict" }

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
    traits [@ReadFrom<["lhs", "rhs"]>, @Alloc<"res">]
    config { fmt = "$lhs $op $rhs attr-dict" }

  Alias @UnaryOp -> #dmc.AnyOf<
      "not", "#", "-", "~">
  Op @unary(val: !lua.value) -> (res: !lua.value)
    { op = #lua.UnaryOp }
    traits [@ReadFrom<"val">, @Alloc<"res">]
    config { fmt = "$op $val attr-dict" }

  Op @numeric_for(lower: !lua.value, upper: !lua.value, step: !lua.value) -> ()
    { ivar = #dmc.String } (region: Sized<1>)
    traits [@ReadFrom<["lower", "upper", "step"]>]
    config { fmt = "$ivar `in` `[` $lower `,` $upper `]` `by` $step `do` $region attr-dict" }
  Op @function_def() -> (fcn: !lua.value)
    { params = #dmc.ArrayOf<#dmc.String> } (region: Sized<1>)
    traits [@MemoryWrite]
    config { fmt = "$params $region attr-dict" }
  Op @function_def_capture(captures: !dmc.Variadic<!lua.value>) -> (fcn: !lua.value)
    { params = #dmc.ArrayOf<#dmc.String> } (region: Sized<1>)
    traits [@MemoryWrite, @SameVariadicOperandSizes]
    config { fmt = "`(` operands `)` `:` type(operands) $params $region attr-dict" }
  Op @cond_if(cond: !lua.value) -> () (first: Sized<1>, second: Sized<1>)
    traits [@ReadFrom<"cond">] config { fmt = "$cond `then` $first `else` $second attr-dict" }
  Op @loop_while() -> () (eval: Sized<1>, region: Sized<1>)
    config { fmt = "$eval `do` $region attr-dict" }
  Op @end() -> () traits [@IsTerminator] config { fmt = "attr-dict" }
  Op @ret(vals: !dmc.Variadic<!lua.value>) -> ()
    traits [@MemoryWrite, @IsTerminator, @SameVariadicOperandSizes, @ReadFrom<"vals">]
    config { fmt = "($vals^ `:` type($vals))? attr-dict" }
  Op @cond(cond: !lua.value) -> () traits [@ReadFrom<"cond">, @IsTerminator]
    config { fmt = "$cond attr-dict" }
}

Dialect @luac {
  Alias @bool -> i1 { builder = "IntegerType(1)" }
  Alias @integer -> i64 { builder = "IntegerType(64)" }
  Alias @real -> f64 { builder = "F64Type()" }
  Alias @pack_fcn -> (!lua.pack, !lua.pack) -> !lua.pack
    { builder = "FunctionType([lua.pack(), lua.pack()], [lua.pack()])" }

  Alias @type_enum -> i16 { builder = "IntegerType(16)" }
  Alias @type_nil -> 0 : i16
  Alias @type_bool -> 1 : i16
  Alias @type_num -> 2 : i16
  Alias @type_str -> 3 : i16
  Alias @type_tbl -> 4 : i16
  Alias @type_fcn -> 5 : i16
  // userdata, thread unimplemented

  Op @wrap_int(num: !luac.integer) -> (res: !lua.value)
    traits [@Alloc<"res">]
    config { fmt = "$num attr-dict" }
  Op @wrap_real(num: !luac.real) -> (res: !lua.value)
    traits [@Alloc<"res">]
    config { fmt = "$num attr-dict" }

  Op @add(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    traits [@ReadFrom<["lhs", "rhs"]>]
    config { fmt = "`(` operands `)` attr-dict" }
  Op @sub(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    traits [@ReadFrom<["lhs", "rhs"]>]
    config { fmt = "`(` operands `)` attr-dict" }
  Op @mul(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    traits [@ReadFrom<["lhs", "rhs"]>]
    config { fmt = "`(` operands `)` attr-dict" }
  Op @eq(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    traits [@ReadFrom<["lhs", "rhs"]>]
    config { fmt = "`(` operands `)` attr-dict" }
  Op @lt(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    traits [@ReadFrom<["lhs", "rhs"]>]
    config { fmt = "`(` operands `)` attr-dict" }
  Op @le(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    traits [@ReadFrom<["lhs", "rhs"]>]
    config { fmt = "`(` operands `)` attr-dict" }
  Op @bool_and(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
   traits [@ReadFrom<["lhs", "rhs"]>]
    config { fmt = "`(` operands `)` attr-dict" }

  Op @alloc() -> (res: !lua.value)
    traits [@Alloc<"res">] config { fmt = "attr-dict" }
  Op @alloc_gc(tgt: !lua.value) -> ()
    traits [@MemoryAlloc, @WriteTo<"tgt">] config { fmt = "$tgt attr-dict" }

  Op @set_type(tgt: !lua.value, ty: !luac.type_enum) -> ()
    traits [@WriteTo<"tgt">]
    config { fmt = "`type` `(` $tgt `)` `=` $ty attr-dict" }
  Op @get_type(tgt: !lua.value) -> (ty: !luac.type_enum)
    traits [@ReadFrom<"tgt">]
    config { fmt = "`type` `(` $tgt `)` attr-dict" }

  Op @set_bool_val(tgt: !lua.value, b: !luac.bool) -> ()
    traits [@WriteTo<"tgt">]
    config { fmt = "$tgt `=` $b attr-dict" }
  Op @get_bool_val(tgt: !lua.value) -> (b: !luac.bool)
    traits [@ReadFrom<"tgt">]
    config { fmt = "$tgt attr-dict" }

  Op @set_int64_val(tgt: !lua.value, num: !luac.integer) -> ()
    traits [@WriteTo<"tgt">]
    config { fmt = "$tgt `=` $num attr-dict" }
  Op @get_int64_val(tgt: !lua.value) -> (num: !luac.integer)
    traits [@ReadFrom<"tgt">]
    config { fmt = "$tgt attr-dict" }

  Op @set_double_val(tgt: !lua.value, num: !luac.real) -> ()
    traits [@WriteTo<"tgt">]
    config { fmt = "$tgt `=` $num attr-dict" }
  Op @get_double_val(tgt: !lua.value) -> (num: !luac.real)
    traits [@ReadFrom<"tgt">]
    config { fmt = "$tgt attr-dict" }

  Op @get_fcn_addr(fcn: !lua.value) -> (fcn_addr: !luac.pack_fcn)
    traits [@ReadFrom<"fcn">]
    config { fmt = "$fcn attr-dict" }
  Op @set_fcn_addr(fcn: !lua.value, fcn_addr: !luac.pack_fcn) -> ()
    traits [@WriteTo<"fcn">]
    config { fmt = "$fcn `=` $fcn_addr attr-dict" }
  Op @get_capture_pack(fcn: !lua.value) -> (pack: !lua.value_pack)
    traits [@ReadFrom<"fcn">]
    config { fmt = "$fcn `[` `]` attr-dict" }
  Op @set_capture_pack(fcn: !lua.value, pack: !lua.value_pack) -> ()
    traits [@WriteTo<"fcn">]
    config { fmt = "$fcn `[` `]` `=` $pack attr-dict" }

  Op @get_value_union(tgt: !lua.value) -> (u: i64)
    traits [@ReadFrom<"tgt">]
    config { fmt = "$tgt attr-dict" }
  Op @set_value_union(tgt: !lua.value, u: i64) -> ()
    traits [@WriteTo<"tgt">]
    config { fmt = "$tgt `=` $u attr-dict" }

  Op @is_int(tgt: !lua.value) -> (ret: !luac.bool)
    traits [@ReadFrom<"tgt">]
    config { fmt = "$tgt attr-dict" }

  Op @new_pack(rsv: i64) -> (pack: !lua.value_pack)
    traits [@Alloc<"pack">]
    config { fmt = "`[` $rsv `]` attr-dict" }
  Op @delete_pack(pack: !lua.value_pack) -> ()
    traits [@Free<"pack">]
    config { fmt = "$pack attr-dict" }
  Op @pack_push(pack: !lua.value_pack, val: !lua.value) -> ()
    traits [@WriteTo<"pack">, @ReadFrom<["pack", "val"]>]
    config { fmt = "`(` $pack `,` `[` $val `]` `)` attr-dict" }
  Op @pack_push_all(pack: !lua.value_pack, vals: !lua.value_pack) -> ()
    traits [@WriteTo<"pack">, @ReadFrom<["pack", "vals"]>]
    config { fmt = "`(` $pack `,` $vals `)` attr-dict" }
  Op @pack_pull_one(pack: !lua.value_pack) -> (val: !lua.value)
    traits [@WriteTo<"pack">, @ReadFrom<"pack">]
    config { fmt = "$pack attr-dict" }
  Op @pack_get_size(pack: !lua.value_pack) -> (sz: i64)
    traits [@ReadFrom<"pack">] config { fmt = "$pack attr-dict" }
  Op @pack_rewind(pack: !lua.value_pack) -> ()
    traits [@WriteTo<"pack">] config { fmt = "$pack attr-dict" }

  Op @global_string() -> () { sym = #dmc.String, value = #dmc.String }
    traits [@MemoryWrite] config { fmt = "symbol($sym) `=` $value attr-dict" }
  Op @load_string() -> (res: !lua.value) { global_sym = #dmc.String }
    traits [@Alloc<"res">] config { fmt = "symbol($global_sym) attr-dict" }
}

Dialect @luallvm {
  Alias @ref -> !llvm<"{ i16, i16, { i64 } }*">
  Alias @pack -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">

  Op @load_string(data: !llvm<"i8*">, length: !llvm.i64) -> (val: !luallvm.ref)
    traits [@Alloc<"val">]
    config { fmt = "`(` operands `)` `:` functional-type(operands, results) attr-dict" }
}
