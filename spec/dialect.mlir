dmc.Dialect @test {
  dmc.Op @op_a(!dmc.AnyInteger, !dmc.AnyOf<!dmc.AnyI<32>, !dmc.AnyFloat>) -> !dmc.UI<32>
      { attr0 = #dmc.APInt }
  dmc.Op @op_b(!dmc.AnyFloat, !dmc.F<16>) -> (!dmc.BF16, !dmc.SI<32>)
      { attr1 = #dmc.Bool }
  dmc.Op @my_ret(!dmc.AnyInteger, !dmc.Variadic<!dmc.Any>) -> ()
      { attr2 = #dmc.Optional<#dmc.Bool> }
      traits [@SameVariadicOperandSizes]
      config { is_terminator = true }

  dmc.Type @CustomType
  dmc.Op @op_c(!test.CustomType) -> !test.CustomType

  dmc.Type @Array2D<i64, i64>
  dmc.Alias @IsArray2D -> !dmc.Isa<@test::@Array2D>
  dmc.Op @transpose(!test.IsArray2D) -> !test.IsArray2D

  dmc.Op @get_value() -> !dmc.Any

  dmc.Attr @CustomAttr
  dmc.Attr @Pair<#dmc.APInt, #dmc.APInt>
  dmc.Alias @IsPair -> #dmc.Isa<@test::@Pair>
  dmc.Alias @IsCustomAttr -> #dmc.Isa<@test::@CustomAttr>
  dmc.Op @op_d() -> () { attr3 = #test.IsPair,
                         attr4 = #test.IsCustomAttr }

  dmc.Attr @Box<#dmc.Any>
  dmc.Attr @CustomPair<#dmc.Isa<@test::@Box>, #dmc.Isa<@test::@Box>>

  dmc.Type @BoxType<#dmc.Isa<@test::@Box>>
  dmc.Op @op_e(!test.BoxType<#test.Box<6>>) -> ()
}
