Dialect @py {
  Type @object
  Alias @obj -> !dmc.Isa<@py::@object> { builder = "py.object()" }



  Op @func() -> ()
    { name = #dmc.String, sig = #dmc.Type }
    (body: Sized<1>)
    traits [@MemoryWrite, @IsIsolatedFromAbove]
    config { fmt = "symbol($name) $sig $body attr-dict" }

  Op @call(func: !py.obj,
           args: !dmc.Variadic<!py.obj>) -> (rets: !py.obj)
    traits [@MemoryWrite, @SameVariadicOperandSizes]
    config { fmt = "$func`(`$args`)` `:` functional-type($args, $rets) attr-dict" }

  Op @ret(args: !dmc.Variadic<!py.obj>) -> ()
    traits [@IsTerminator, @SameVariadicOperandSizes]
    config { fmt = "($args^ `:` type($args))? attr-dict" }



  Type @handle
  Alias @ref -> !dmc.Isa<@py::@handle> { builder = "py.handle()" }

  Op @name() -> (ref: !py.ref)
    { var = #dmc.String }
    config { fmt = "$var attr-dict" }

  Op @assign(ref: !py.ref, arg: !py.obj) -> (new: !py.ref)
    traits [@WriteTo<"ref">]
    config { fmt = "$ref `=` $arg attr-dict" }

  Op @load(ref: !py.ref) -> (res: !py.obj)
    traits [@NoSideEffects]
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
Dialect @stencil {
  /// Base constraints.
  Alias @Shape -> #dmc.AllOf<#dmc.Array, #dmc.ArrayOf<#dmc.APInt>>
  Alias @ArrayCount3 -> #dmc.Py<"isinstance({self}, ArrayAttr) and len({self}) == 3">

  /// Stencil types: FieldType and TempType, both subclass GridType.
  Type @field<shape: #stencil.Shape, type: #dmc.Type>
    { fmt = "`<` dims($shape) $type `>`" }
  Type @temp <shape: #stencil.Shape, type: #dmc.Type>
    { fmt = "`<` dims($shape) $type `>`" }
  Alias @Field -> !dmc.Isa<@stencil::@field>
  Alias @Temp  -> !dmc.Isa<@stencil::@temp>

  /// Element type and index attribute constraints.
  Alias @None -> !dmc.None { builder = "NoneType()" }
  Alias @Element -> !dmc.AnyOf<f32, f64>
  Alias @Index -> #dmc.AllOf<#dmc.ArrayOf<#dmc.APInt>, #stencil.ArrayCount3>
    { type = !stencil.None }
  Alias @OptionalIndex -> #dmc.Optional<#stencil.Index>
    { type = !stencil.None }

  /// AssertOp
  Op @assert(field: !stencil.Field) -> () { lb = #stencil.Index,
                                            ub = #stencil.Index }
    traits [@MemoryWrite]
    config { fmt = "$field `(` $lb `:` $ub `)` attr-dict-with-keyword `:` type($field)" }

  /// AccessOp
  Op @access(temp: !stencil.Temp) -> (res: !stencil.Element)
    { offset = #stencil.Index }
    config { fmt = "$temp $offset attr-dict-with-keyword `:` functional-type($temp, $res)" }

  /// LoadOp
  Op @load(field: !stencil.Field) -> (res: !stencil.Temp)
    { lb = #stencil.OptionalIndex, ub = #stencil.OptionalIndex }
    config { fmt = "$field (`(` $lb^ `:` $ub `)`)? attr-dict-with-keyword `:` functional-type($field, $res)" }

  /// StoreOp
  Op @store(temp: !stencil.Temp, field: !stencil.Field) -> ()
    { lb = #stencil.Index, ub = #stencil.Index }
    traits [@WriteTo<"field">]
    config { fmt = "$temp `to` $field `(` $lb `:` $ub `)` attr-dict-with-keyword `:` type($temp) `to` type($field)" }

  /// ApplyOp
  Op @apply(operands: !dmc.Variadic<!dmc.Any>) -> (res: !dmc.Variadic<!stencil.Temp>)
    { lb = #stencil.OptionalIndex, ub = #stencil.OptionalIndex }
    (region: Sized<1>)
    traits [@SameVariadicOperandSizes, @SameVariadicResultSizes,
            @IsIsolatedFromAbove,
            @SingleBlockImplicitTerminator<"stencil.return">]
    config { is_isolated_from_above = true,
             fmt = "`(` $operands `)` `:` functional-type($operands, $res) attr-dict-with-keyword $region (`to` `(` $lb^ `:` $ub `)`)?" }

  /// ReturnOp
  Op @return(operands: !dmc.Variadic<!stencil.Element>) -> ()
    { unroll = #stencil.OptionalIndex }
    traits [@SameVariadicOperandSizes, @HasParent<"stencil.apply">, @IsTerminator]
    config { fmt = "(`unroll` $unroll^)? $operands attr-dict-with-keyword `:` type($operands)" }
}
Dialect @tmp {
  Op @stencil_module() -> (res: !py.obj)
    traits [@NoSideEffects]
  Op @stencil_assert() -> (res: !py.obj)
    traits [@NoSideEffects]
  Op @stencil_load() -> (res: !py.obj)
    traits [@NoSideEffects]
  Op @stencil_store() -> (res: !py.obj)
    traits [@NoSideEffects]
  Op @stencil_apply() -> (res: !py.obj)
    traits [@NoSideEffects]

  Op @stencil_index() -> (res: !py.obj)
    { index = #dmc.Any }
    traits [@NoSideEffects]
  Op @stencil_apply_body() -> (res: !py.obj)
    (body: Sized<1>)
    traits [@NoSideEffects]

  Alias @unshaped_f64_field -> !stencil.field<?x?x?xf64>
  Alias @unshaped_f32_field -> !stencil.field<?x?x?xf32>
  Alias @unshaped_f64_temp -> !stencil.temp<?x?x?xf64>
  Alias @unshaped_f32_temp -> !stencil.temp<?x?x?xf32>
}
