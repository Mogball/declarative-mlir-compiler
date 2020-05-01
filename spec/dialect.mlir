dmc.Dialect @MyDialect attributes {
  allowsUnregisteredOps = false,
  foldHook = @myPythonFoldHook
} {
  dmc.Op @MyOpA(!dmc.Int, !dmc.Any, f32) -> (!dmc.Float)
      { commutative = true, traits = [@SameOperandTypes, @myPythonTrait],
        attrs = {
          attr0 = #dmc.Int,
          attr1 = #dmc.Typed<!dmc.Any<f32, !dmc.Int>>,
          attr2 = #dmc.Optional<#dmc.Any>,
          attr3 = #dmc.Type
        }
      }
  dmc.Op @MyOpB(!dmc.AnyOf<i32, i64, !dmc.Float>,
                !dmc.Variadic<!dmc.Int>,
                !dmc.Variadic<!dmc.Float>) -> (!dmc.Any)
      { terminator = true, traits = [@AttrSizedOperandSegments] }
}
