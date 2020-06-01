dmc.Dialect @test {
  dmc.Op @op_a(arg0 : !dmc.AnyInteger, arg1 : !dmc.AnyOf<!dmc.AnyI<32>, !dmc.AnyFloat>) -> (ret0 : !dmc.UI<32>)
      { attr0 = #dmc.APInt }
  dmc.Op @op_b(arg0 : !dmc.AnyFloat, arg1 : !dmc.F<16>) -> (ret0 : !dmc.BF16, ret1 : !dmc.SI<32>)
      { attr1 = #dmc.Bool }
  dmc.Op @my_ret(arg0 : !dmc.AnyInteger, arg1 : !dmc.Variadic<!dmc.Any>) -> ()
      { attr2 = #dmc.Optional<#dmc.Bool> }
      traits [@SameVariadicOperandSizes, @AtLeastNOperands<1>]
      config { is_terminator = true }

  dmc.Type @CustomType
  dmc.Op @op_c(arg0 : !test.CustomType) -> (ret0 : !test.CustomType)
      traits [@HasParent<"func">]

  dmc.Type @Array2D<i64, i64>
  dmc.Alias @IsArray2D -> !dmc.Isa<@test::@Array2D>
  dmc.Op @transpose(arg0 : !test.IsArray2D) -> (ret0 : !test.IsArray2D)

  dmc.Op @get_value() -> (value : !dmc.Any)

  dmc.Attr @CustomAttr
  dmc.Attr @Pair<#dmc.APInt, #dmc.APInt>
  dmc.Alias @IsPair -> #dmc.Isa<@test::@Pair>
  dmc.Alias @IsCustomAttr -> #dmc.Isa<@test::@CustomAttr>
  dmc.Op @op_d() -> () { attr3 = #test.IsPair,
                         attr4 = #test.IsCustomAttr }

  dmc.Attr @Box<#dmc.Any>
  dmc.Attr @CustomPair<#dmc.Isa<@test::@Box>, #dmc.Isa<@test::@Box>>

  dmc.Type @BoxType<#dmc.Isa<@test::@Box>>
  dmc.Op @op_e(arg0 : !test.BoxType<#test.Box<6>>) -> ()

  dmc.Op @op_regions() -> () {} (Any, Sized<2>, Variadic<IsolatedFromAbove>)
  dmc.Op @ret() -> () config { is_terminator = true }

  dmc.Alias @IsInteger -> !dmc.Py<"isinstance({self}, IntegerType)">
  dmc.Alias @ArraySize3 -> #dmc.Py<"isinstance({self}, ArrayAttr) and len({self}) == 3">
  dmc.Op @op_py(arg0 : !test.IsInteger) -> () { index = #test.ArraySize3 }
}
