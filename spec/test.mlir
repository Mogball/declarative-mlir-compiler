module {
  func @test0(%arg0 : i32, %arg1 : i64, %arg2 : f16) -> (bf16) {
    %0 = "test.op_a"(%arg0, %arg2) { attr0 = 6 } : (i32, f16) -> ui32
    %1, %2 = "test.op_b"(%arg2, %arg2) { attr1 = true } : (f16, f16) -> (bf16, si32)
    "test.my_ret"(%2, %0, %1, %arg1) { attr2 = false } : (si32, ui32, bf16, i64) -> ()
  }
}
