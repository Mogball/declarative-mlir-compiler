dmc.Dialect @MyDialect {
  allowsUnregisteredOps = false,
  foldHook = @myPythonFoldHook
} {
  dmc.Op @MyOpA(%0 : !dmc.Int, %1 : !dmc.Any, %2 : f32) -> !dmc.Float
      { commutative = true, traits = [@SameOperandTypes, @myPythonTrait] }
  dmc.Op @MyOpB(%0 : !dmc.AnyOf<i32, i64, !dmc.Float>,
                %1 : !dmc.Variadic<!dmc.Int>,
                %2 : !dmc.Variadic<!dmc.Float>) -> !dmc.NoneOf<f32>
      { terminator = true, traits = [@SameVariadicSizes] }
}
