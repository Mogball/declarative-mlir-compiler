dmc.Dialect @stencil {
  /// Base constraints.
  dmc.Alias @Shape -> #dmc.AllOf<#dmc.Array, #dmc.ArrayOf<#dmc.APInt>>
  dmc.Alias @ArrayCount3 -> #dmc.Py<"isinstance({self}, ArrayAttr) and len({self}) == 3">

  /// Stencil types: FieldType and TempType, both subclass GridType.
  dmc.Type @field<#stencil.Shape, #dmc.Type>
  dmc.Type @temp <#stencil.Shape, #dmc.Type>
  dmc.Alias @Field -> !dmc.Isa<@stencil::@field>
  dmc.Alias @Temp  -> !dmc.Isa<@stencil::@temp>

  /// Element type and index attribute constraints.
  dmc.Alias @None -> !dmc.None { builder = "NoneType()" }
  dmc.Alias @Element -> !dmc.AnyOf<f32, f64>
  dmc.Alias @Index -> #dmc.AllOf<#dmc.ArrayOf<#dmc.APInt>, #stencil.ArrayCount3>
    { type = !stencil.None }
  dmc.Alias @OptionalIndex -> #dmc.Optional<#stencil.Index>
    { type = !stencil.None }

  /// AssertOp
  dmc.Op @assert(field : !stencil.Field) -> () { lb = #stencil.Index,
                                                 ub = #stencil.Index }
    config { fmt = "$field `(` $lb `:` $ub `)` attr-dict-with-keyword `:` type($field)" }

  /// AccessOp
  dmc.Op @access(temp : !stencil.Temp) -> (res : !stencil.Element)
    { offset = #stencil.Index }
    config { fmt = "$temp $offset attr-dict-with-keyword `:` functional-type($temp, $res)" }

  /// LoadOp
  dmc.Op @load(field : !stencil.Field) -> (res : !stencil.Temp)
    { lb = #stencil.OptionalIndex, ub = #stencil.OptionalIndex }
    config { fmt = "$field (`(` $lb^ `:` $ub `)`)? attr-dict-with-keyword `:` functional-type($field, $res)" }

  /// StoreOp
  dmc.Op @store(temp : !stencil.Temp, field : !stencil.Field) -> ()
    { lb = #stencil.Index, ub = #stencil.Index }
    config { fmt = "$temp `to` $field `(` $lb `:` $ub `)` attr-dict-with-keyword `:` type($temp) `to` type($field)" }

  /// ApplyOp
  dmc.Op @apply(operands : !dmc.Variadic<!dmc.Any>) -> (res : !dmc.Variadic<!stencil.Temp>)
    { lb = #stencil.OptionalIndex, ub = #stencil.OptionalIndex }
    (region : Sized<1>)
    traits [@SameVariadicOperandSizes, @SameVariadicResultSizes]
    config { is_isolated_from_above = true }
             //fmt = "`(` $operands `:` type($operands) `)` `->` type($res) attr-dict-with-keyword (`to` `(` $lb^ `:` $ub `)`)?" }

  /// ReturnOp
  dmc.Op @return(operands : !dmc.Variadic<!stencil.Element>) -> ()
    { unroll = #stencil.OptionalIndex }
    traits [@SameVariadicOperandSizes]
    config { is_terminator = true,
             fmt = "(`unroll` $unroll^)? $operands attr-dict-with-keyword `:` type($operands)" }
}
