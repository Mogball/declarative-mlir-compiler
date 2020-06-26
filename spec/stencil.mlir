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
    config { fmt = "$temp `to` $field `(` $lb `:` $ub `)` attr-dict-with-keyword `:` type($temp) `to` type($field)" }

  /// ApplyOp
  Op @apply(operands: !dmc.Variadic<!dmc.Any>) -> (res: !dmc.Variadic<!stencil.Temp>)
    { lb = #stencil.OptionalIndex, ub = #stencil.OptionalIndex }
    (region: Sized<1>)
    traits [@SameVariadicOperandSizes, @SameVariadicResultSizes,
            @SingleBlockImplicitTerminator<"stencil.return">]
    config { is_isolated_from_above = true,
             fmt = "`(` $operands `)` `:` functional-type($operands, $res) attr-dict-with-keyword $region (`to` `(` $lb^ `:` $ub `)`)?" }

  /// ReturnOp
  Op @return(operands: !dmc.Variadic<!stencil.Element>) -> ()
    { unroll = #stencil.OptionalIndex }
    traits [@SameVariadicOperandSizes, @HasParent<"stencil.apply">, @IsTerminator]
    config { fmt = "(`unroll` $unroll^)? $operands attr-dict-with-keyword `:` type($operands)" }
}
