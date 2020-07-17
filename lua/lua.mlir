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
    { params = #dmc.ArrayOf<#dmc.String> } (region: Any)
    traits [@SameVariadicOperandSizes, @NoSideEffects, @IsIsolatedFromAbove]
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
  Op @cond_ret(pack: !lua.value_pack) -> ()
    traits [@IsTerminator, @HasParent<"lua.function_def_capture">]

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
    traits [@NoSideEffects]
  Op @table_set_prealloc(tbl: !lua.value, iv: i64, val: !lua.value) -> ()
    traits [@WriteTo<"tbl">]

  Op @unpack_unsafe(pack: !lua.value_pack) -> (vals: !dmc.Variadic<!lua.value>)
    traits [@SameVariadicResultSizes, @NoSideEffects]

  Op @pack_func(captures: !dmc.Variadic<!lua.value>) -> (fcn: !lua.value)
    (region: Any)
    traits [@NoSideEffects, @SameVariadicOperandSizes, @IsIsolatedFromAbove]
}

Dialect @luac {
  Alias @_int -> i64 { builder = "IntegerType(64)" }

  Alias @bool -> i1 { builder = "IntegerType(1)" }
  Alias @real -> f64 { builder = "F64Type()" }
  Alias @pack_fcn -> (!lua.capture, !lua.pack) -> !lua.pack
    { builder = "FunctionType([lua.capture(), lua.pack()], [lua.pack()])" }

  Alias @type_enum -> i32 { builder = "IntegerType(32)" }
  Alias @type_nil -> 0 : i32
  Alias @type_bool -> 1 : i32
  Alias @type_num -> 2 : i32
  Alias @type_str -> 3 : i32
  Alias @type_tbl -> 4 : i32
  Alias @type_fcn -> 5 : i32
  // userdata, thread unimplemented

  Op @wrap_bool(b: !luac.bool) -> (res: !lua.value)
    traits [@Alloc<"res">]
    config { fmt = "$b attr-dict" }
  Op @wrap_real(num: !luac.real) -> (res: !lua.value)
    traits [@Alloc<"res">]
    config { fmt = "$num attr-dict" }
  Op @make_fcn(addr: !luac.pack_fcn, capture: !lua.capture_pack) -> (fcn: !lua.value)
    traits [@Alloc<"fcn">]

  /// Binary operations
  Op @add(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    traits [@Alloc<"res">]
    config { fmt = "`(` operands `)` attr-dict" }
  Op @sub(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    traits [@Alloc<"res">]
    config { fmt = "`(` operands `)` attr-dict" }
  Op @mul(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    traits [@Alloc<"res">]
    config { fmt = "`(` operands `)` attr-dict" }
  Op @pow(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    traits [@Alloc<"res">]
    config { fmt = "`(` operands `)` attr-dict" }
  Op @strcat(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    traits [@Alloc<"res">]
    config { fmt = "`(` operands `)` attr-dict" }
  Op @eq(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    traits [@Alloc<"res">]
    config { fmt = "`(` operands `)` attr-dict" }
  Op @ne(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    traits [@Alloc<"res">]
    config { fmt = "`(` operands `)` attr-dict" }
  Op @lt(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    traits [@Alloc<"res">]
    config { fmt = "`(` operands `)` attr-dict" }
  Op @le(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    traits [@Alloc<"res">]
    config { fmt = "`(` operands `)` attr-dict" }
  Op @ge(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    traits [@Alloc<"res">]
    config { fmt = "`(` operands `)` attr-dict" }
  Op @gt(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    traits [@Alloc<"res">]
    config { fmt = "`(` operands `)` attr-dict" }
  Op @bool_and(lhs: !lua.value, rhs: !lua.value) -> (res: !lua.value)
    traits [@Alloc<"res">]
    config { fmt = "`(` operands `)` attr-dict" }

  /// Unary operations
  Op @bool_not(val: !lua.value) -> (res: !lua.value)
    traits [@Alloc<"res">]
    config { fmt = "$val attr-dict" }
  Op @list_size(val: !lua.value) -> (res: !lua.value)
    traits [@Alloc<"res">]
    config { fmt = "$val attr-dict" }
  Op @neg(val: !lua.value) -> (res: !lua.value)
    traits [@Alloc<"res">]
    config { fmt = "$val attr-dict" }

  /// Misc library functions
  Type @void_ptr
  Alias @impl_ptr -> !dmc.Isa<@luac::@void_ptr> { builder = "luac.void_ptr()" }
  Op @get_impl(val: !lua.value) -> (impl: !luac.impl_ptr)
    config { fmt = "$val attr-dict" }

  Op @convert_bool_like(val: !lua.value) -> (b: !luac.bool)
    config { fmt = "$val attr-dict" }

  /// Simple value Manipulation
  Op @get_type(val: !lua.value) -> (ty: !luac.type_enum)
    config { fmt = "`type` `(` $val `)` attr-dict" }
  Op @get_bool_val(val: !lua.value) -> (b: !luac.bool)
    config { fmt = "$val attr-dict" }
  Op @get_double_val(val: !lua.value) -> (num: !luac.real)
    config { fmt = "$val attr-dict" }
  Op @get_fcn_addr(val: !lua.value) -> (addr: !luac.pack_fcn)
    config { fmt = "$val attr-dict" }
  Op @get_capture_pack(val: !lua.value) -> (capture: !lua.capture_pack)
    config { fmt = "$val `[` `]` attr-dict" }

  Op @new_capture(size: i32) -> (capture: !lua.capture_pack)
  Op @add_capture(capture: !lua.capture_pack, val: !lua.value, idx: i32) -> ()
    traits [@WriteTo<"capture">]
  Op @get_capture(capture: !lua.capture_pack, idx: i32) -> (val: !lua.value)
    traits [@NoSideEffects]

  Op @get_arg_pack(size: i32) -> (pack: !lua.value_pack)
  Op @get_ret_pack(size: i32) -> (pack: !lua.value_pack)

  Op @pack_insert(pack: !lua.value_pack, val: !lua.value, idx: i32) -> ()
    traits [@WriteTo<"pack">]
    config { fmt = "$pack `[` $idx `]` `=` $val attr-dict" }
  Op @pack_insert_all(pack: !lua.value_pack, tail: !lua.value_pack, idx: i32) -> ()
    traits [@WriteTo<"pack">]
  Op @pack_get(pack: !lua.value_pack, idx: i32) -> (res: !lua.value)
  Op @pack_get_unsafe(pack: !lua.value_pack, idx: i32) -> (res: !lua.value)
    config { fmt = "$pack `[` $idx `]` attr-dict" }
  Op @pack_get_size(pack: !lua.value_pack) -> (size: i32)
    config { fmt = "$pack attr-dict" }

  Op @global_string() -> () { sym = #dmc.String, value = #dmc.String }
    traits [@MemoryWrite] config { fmt = "symbol($sym) `=` $value attr-dict" }
  Op @load_string() -> (res: !lua.value) { global_sym = #dmc.String }
    config { fmt = "symbol($global_sym) attr-dict" }

  Op @into_alloca(val: !lua.value) -> (res: !lua.value)
    config { fmt = "$val attr-dict" }
  Op @load_from(val: !lua.value) -> (res: !lua.value)
    config { fmt = "$val attr-dict" }
}

Dialect @luallvm {
  /// TODO auto-generated builders would be nice
  Alias @value    -> !llvm<"{ i32, i64 }">  { builder = "LLVMType.Struct([LLVMType.Int32(), LLVMType.Int64()])" }
  Alias @ref      -> !llvm<"{ i32, i64 }*"> { builder = "LLVMType.Struct([LLVMType.Int32(), LLVMType.Int64()]).ptr_to()" }
  Alias @type     -> !llvm.i32              { builder = "LLVMType.Int32()" }
  Alias @u        -> !llvm.i64              { builder = "LLVMType.Int64()" }
  Alias @impl     -> !llvm<"i8*">           { builder = "LLVMType.Int8Ptr()" }

  Alias @pack     -> !llvm<"{ i32, { i32, i64 }* }">  { builder = "LLVMType.Struct([LLVMType.Int32(), luallvm.ref()])" }
  Alias @capture  -> !llvm<"{ i32, i64 }**">          { builder = "luallvm.ref().ptr_to()" }
  Alias @fcn      -> !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*">  { builder = "LLVMType.Func(luallvm.pack(), [luallvm.capture(), luallvm.pack()])" }
  Alias @closure  -> !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }">  { builder = "LLVMType.Struct([luallvm.fcn(), luallvm.capture()])" }

  Alias @closure_ptr  -> !llvm<"{ { i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })*, { i32, i64 }** }*">
  Alias @capture_ptr  -> !llvm<"{ i32, i64 }***">
  Alias @fcn_ptr      -> !llvm<"{ i32, { i32, i64 }* } ({ i32, i64 }**, { i32, { i32, i64 }* })**">

  Op @alloca_value() -> (ref: !luallvm.ref)
    traits [@Alloc<"ref">] config { fmt = "attr-dict" }
  Op @load_builtin() -> (val: !luallvm.value) { builtin = #dmc.String }
    traits [@NoSideEffects] config { fmt = "symbol($builtin) attr-dict" }

  Op @const_type() -> (type: !luallvm.type) { value = #dmc.I<32> }
    traits [@NoSideEffects] config { fmt = "$value attr-dict" }
  Op @get_type_direct(ref: !luallvm.ref) -> (type: !luallvm.type)
    traits [@ReadFrom<"ref">] config { fmt = "$ref attr-dict" }
  Op @set_type_direct(ref: !luallvm.ref, type: !luallvm.type) -> ()
    traits [@WriteTo<"ref">] config { fmt = "$ref `type` `=` $type attr-dict" }
  Op @get_u_direct(ref: !luallvm.ref) -> (u: !luallvm.u)
    traits [@ReadFrom<"ref">] config { fmt = "$ref attr-dict" }
  Op @set_u_direct(ref: !luallvm.ref, u: !luallvm.u) -> ()
    traits [@WriteTo<"ref">] config { fmt = "$ref `u` `=` $u attr-dict" }
  Op @get_impl_direct(ref: !luallvm.ref) -> (impl: !luallvm.impl)
    traits [@ReadFrom<"ref">] config { fmt = "$ref attr-dict" }
  Op @set_impl_direct(ref: !luallvm.ref, impl: !luallvm.impl) -> ()
    traits [@WriteTo<"ref">] config { fmt = "$ref `impl` `=` $impl attr-dict" }

  Op @new_table_impl() -> (impl: !luallvm.impl)
    traits [@Alloc<"impl">] config { fmt = "attr-dict" }
  Op @get_string_data() -> (data: !llvm<"i8*">, length: !llvm.i64) { sym = #dmc.String }
    traits [@NoSideEffects] config { fmt = "symbol($sym) attr-dict" }
  Op @load_string_impl(data: !llvm<"i8*">, length: !llvm.i64) -> (impl: !luallvm.impl)
    traits [@Alloc<"impl">] config { fmt = "`(` operands `)` `:` type(operands) attr-dict" }
  Op @make_fcn_impl(addr: !luallvm.fcn,
                    capture: !luallvm.capture) -> (impl: !luallvm.impl)
    traits [@Alloc<"impl">] config { fmt = "`(` operands `)` attr-dict" }

  Op @table_get_impl(impl: !luallvm.impl, key: !luallvm.value) -> (val: !luallvm.value)
    traits [@ReadFrom<"impl">] config { fmt = "$impl `[` $key `]` attr-dict" }
  Op @table_set_impl(impl: !luallvm.impl, key: !luallvm.value, val: !luallvm.value) -> ()
    traits [@WriteTo<"impl">] config { fmt = "$impl `[` $key `]` `=` $val attr-dict" }

  Op @table_get_prealloc_impl(impl: !luallvm.impl, iv: i64) -> (val: !luallvm.value)
    traits [@ReadFrom<"impl">] config { fmt = "$impl `[` $iv `]` attr-dict" }
  Op @table_set_prealloc_impl(impl: !luallvm.impl, iv: i64, val: !luallvm.value) -> ()
    traits [@WriteTo<"impl">] config { fmt = "$impl `[` $iv `]` `=` $val attr-dict" }

  Alias @type_ptr -> !llvm<"i32*">          { builder = "LLVMType.Int32().ptr_to()" }
  Alias @u_ptr    -> !llvm<"i64*">          { builder = "LLVMType.Int64().ptr_to()" }
  Alias @impl_ptr -> !llvm<"i8**">          { builder = "LLVMType.Int8Ptr().ptr_to()" }

  Op @get_type_ptr(ref: !luallvm.ref) -> (type_ptr: !luallvm.type_ptr)
    traits [@ReadFrom<"ref">] config { fmt = "$ref attr-dict" }
  Op @get_type(type_ptr: !luallvm.type_ptr) -> (type: !luallvm.type)
    traits [@ReadFrom<"type_ptr">] config { fmt = "$type_ptr attr-dict" }
  Op @set_type(type_ptr: !luallvm.type_ptr, type: !luallvm.type) -> ()
    traits [@WriteTo<"type_ptr">] config { fmt = "$type_ptr `=` $type attr-dict" }

  Op @get_u_ptr(ref: !luallvm.ref) -> (u_ptr: !luallvm.u_ptr)
    traits [@ReadFrom<"ref">] config { fmt = "$ref attr-dict" }
  Op @get_u(u_ptr: !luallvm.u_ptr) -> (u: !luallvm.u)
    traits [@ReadFrom<"u_ptr">] config { fmt = "$u_ptr attr-dict" }
  Op @set_u(u_ptr: !luallvm.u_ptr, u: !luallvm.u) -> ()
    traits [@WriteTo<"u_ptr">] config { fmt = " $u_ptr `=` $u attr-dict" }

  Op @u_ptr_to_impl_ptr(u_ptr: !luallvm.u_ptr) -> (impl_ptr: !luallvm.impl_ptr)
    traits [@NoSideEffects] config { fmt = "$u_ptr attr-dict" }
  Op @get_impl(impl_ptr: !luallvm.impl_ptr) -> (impl: !luallvm.impl)
    traits [@ReadFrom<"impl_ptr">] config { fmt = "$impl_ptr attr-dict" }
  Op @set_impl(impl_ptr: !luallvm.impl_ptr, impl: !luallvm.impl) -> ()
    traits [@WriteTo<"impl_ptr">] config { fmt = "$impl_ptr `=` $impl attr-dict" }

}
