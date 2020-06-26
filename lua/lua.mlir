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
    traits [@SameVariadicOperandSizes]
    config { fmt = "`(` $vals `)` `:` functional-type($vals, $pack) attr-dict" }
  Op @unpack(pack: !lua.value_pack) -> (vals: !dmc.Variadic<!lua.value>)
    traits [@SameVariadicResultSizes]
    config { fmt = "$pack `:` functional-type($pack, $vals) attr-dict" }

  // Variable handling
  Op @get_or_alloc() -> (res: !lua.value) { var = #dmc.String }
    config { fmt = "$var attr-dict" }
  Op @alloc() -> (res: !lua.value) config { fmt = "attr-dict" }
  Op @assign(tgt: !lua.value, val: !dmc.Any) -> ()
    config { fmt = "$tgt `=` $val `:` type($val) attr-dict" }

  // Function calls
  Op @call(fcn: !lua.value, args: !lua.value_pack) -> (rets: !lua.value_pack)
    traits [@MemoryWrite]
    config { fmt = "$fcn `(` $args `)` attr-dict" }

  // Value getters
  Op @nil() -> (res: !lua.value) config { fmt = "attr-dict" }
  Op @number() -> (res: !lua.value) { value = #dmc.AnyOf<#dmc.AnyI<64>, #dmc.F<64>> }
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
}
