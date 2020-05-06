dmc.Dialect @test {
  dmc.Op @op_a(!dmc.AnyInteger, !dmc.AnyOf<!dmc.AnyI<32>, !dmc.AnyFloat>) -> !dmc.UI<32>
      { attr0 = #dmc.APInt }
  dmc.Op @op_b(!dmc.AnyFloat, !dmc.F<16>) -> (!dmc.BF16, !dmc.SI<32>)
      { attr1 = #dmc.Bool }
  dmc.Op @my_ret(!dmc.AnyInteger, !dmc.Variadic<!dmc.Any>) -> ()
      { attr2 = #dmc.Optional<#dmc.Bool> }
      config { is_terminator = true, traits = [@SameVariadicOperandSizes] }
}
