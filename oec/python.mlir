Dialect @py {
  Type @object
  Alias @obj -> !dmc.Isa<@py::@object> { builder = "py.object()" }



  Op @func() -> ()
    { name = #dmc.String, sig = #dmc.Type }
    (body: Sized<1>)
    traits [@MemoryWrite]
    config { fmt = "symbol($name) $sig $body attr-dict" }

  Op @call(func: !py.obj,
           args: !dmc.Variadic<!py.obj>) -> (rets: !py.obj)
    traits [@NoSideEffects, @SameVariadicOperandSizes]
    config { fmt = "$func`(`$args`)` `:` functional-type($args, $rets) attr-dict" }

  Op @ret(args: !dmc.Variadic<!py.obj>) -> ()
    traits [@IsTerminator, @SameVariadicOperandSizes]
    config { fmt = "($args^ `:` type($args))? attr-dict" }



  Type @handle
  Alias @ref -> !dmc.Isa<@py::@handle> { builder = "py.handle()" }

  Op @name() -> (ref: !py.ref)
    { var = #dmc.String }
    config { fmt = "$var attr-dict" }

  Op @load(ref: !py.ref) -> (res: !py.obj)
    traits [@ReadFrom<"ref">]
    config { fmt = "$ref attr-dict" }

  Op @store(ref: !py.ref, arg: !py.obj) -> ()
    traits [@WriteTo<"ref">]
    config { fmt = "$ref `=` $arg attr-dict" }



  Op @attribute(arg: !py.obj) -> (res: !py.obj)
    { name = #dmc.String }
    traits [@NoSideEffects]
    config { fmt = "$arg`[`$name`]` attr-dict" }

  Op @subscript(arg: !py.obj, idx: !py.obj) -> (res: !py.obj)
    traits [@NoSideEffects]
    config { fmt = "$arg`[`$idx`]` attr-dict" }

  Op @index(arg: !py.obj) -> (res: !py.obj)
    traits [@NoSideEffects]
    config { fmt = "$arg attr-dict" }

  Op @constant() -> (res: !py.obj)
    { value = #dmc.Any }
    traits [@NoSideEffects]
    config { fmt = "$value attr-dict" }

  Op @make_tuple(elts: !dmc.Variadic<!py.obj>) -> (res: !py.obj)
    traits [@NoSideEffects, @SameVariadicOperandSizes]
    config { fmt = "`(`$elts`)` `:` type($elts) attr-dict" }

  Op @make_list(elts: !dmc.Variadic<!py.obj>) -> (res: !py.obj)
    traits [@NoSideEffects, @SameVariadicOperandSizes]
    config { fmt = "`[`$elts`]` `:` type($elts) attr-dict" }



  Alias @UnaryOp -> #dmc.AnyOf<"-">
  Op @unary(arg: !py.obj) -> (res: !py.obj)
    { op = #py.UnaryOp }
    traits [@NoSideEffects]
    config { fmt = "$op $arg attr-dict" }

  Alias @BinOp -> #dmc.AnyOf<"+", "*">
  Op @binary(lhs: !py.obj, rhs: !py.obj) -> (res: !py.obj)
    { op = #py.BinOp }
    traits [@NoSideEffects]
    config { fmt = "$lhs $op $rhs attr-dict" }
}
