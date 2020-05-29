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
  dmc.Alias @Element -> !dmc.AnyOf<f32, f64>
  dmc.Alias @Index -> #dmc.AllOf<#dmc.ArrayOf<#dmc.APInt>, #stencil.ArrayCount3>
  dmc.Alias @OptionalIndex -> #dmc.Optional<#stencil.Index>

  /// AssertOp
  dmc.Op @assert(!stencil.Field) -> () { lb = #stencil.Index,
                                         ub = #stencil.Index }

  /// AccessOp
  dmc.Op @access(!stencil.Temp) -> (!stencil.Element) { offset = #stencil.Index }

  /// LoadOp
  dmc.Op @load(!stencil.Field) -> (!stencil.Temp) { lb = #stencil.OptionalIndex,
                                                    ub = #stencil.OptionalIndex }

  /// StoreOp
  dmc.Op @store(!stencil.Temp, !stencil.Field) -> () { lb = #stencil.Index,
                                                       ub = #stencil.Index }

  /// ApplyOp
  dmc.Op @apply(!dmc.Variadic<!dmc.Any>) -> (!dmc.Variadic<!stencil.Temp>)
      { lb = #stencil.OptionalIndex, ub = #stencil.OptionalIndex }
      (Sized<1>)
      traits [@SameVariadicOperandSizes, @SameVariadicResultSizes]
      config { is_isolated_from_above = true }

  /// ReturnOp
  dmc.Op @return(!stencil.Element) -> () { unroll = #stencil.OptionalIndex }
      config { is_terminator = true }
}
