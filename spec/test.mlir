module {
  func @test0(%arg0 : i32, %arg1 : i64, %arg2 : f16) -> (bf16) {
    %0 = "test.op_a"(%arg0, %arg2) { attr0 = 6 } : (i32, f16) -> ui32
    %1, %2 = "test.op_b"(%arg2, %arg2) { attr1 = true } : (f16, f16) -> (bf16, si32)
    "test.my_ret"(%2, %0, %1, %arg1) { attr2 = false } : (si32, ui32, bf16, i64) -> ()
  }
  func @test1() -> !test.Array2D<3,2> {
    %0 = "test.get_value"() : () -> !test.Array2D<2,3>
    %1 = "test.get_value"() : () -> i32
    %2 = "test.transpose"(%0) : (!test.Array2D<2,3>) -> !test.Array2D<3,2>
    "test.my_ret"(%1, %2) : (i32, !test.Array2D<3,2>) -> ()

    //%0 = "test.get_value"() : () -> !test.CustomType
    //%1 = "test.op_c"(%0) : (!test.CustomType) -> !test.CustomType
    //%2 = "test.get_value"() : () -> i32
    //"test.my_ret"(%2) : (i32) -> ()
  }
}
