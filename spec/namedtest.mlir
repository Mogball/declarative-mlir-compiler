dmc.Dialect @stencil {
  /// Base constraints.
  dmc.Alias @Shape -> #dmc.AllOf<#dmc.Array, #dmc.ArrayOf<#dmc.APInt>>
  dmc.Alias @ArrayCount3 -> #dmc.Py<"isinstance({self}, ArrayAttr) and len({self}) == 3">

  /// Stencil types: FieldType and TempType, both subclass GridType.
  dmc.Type @temp <#stencil.Shape, #dmc.Type>
  dmc.Alias @Temp -> !dmc.Isa<@stencil::@temp>

  /// Element type and index attribute constraints.
  dmc.Alias @Element -> !dmc.AnyOf<f32, f64>
  dmc.Alias @None -> !dmc.None { builder = "NoneType()" }
  dmc.Alias @Index -> #dmc.AllOf<#dmc.ArrayOf<#dmc.APInt>, #stencil.ArrayCount3>
      { type = !stencil.None }

  /// AccessOp
  dmc.Op @access(temp : !stencil.Temp) -> (res : !stencil.Element) { offset = #stencil.Index }
}
