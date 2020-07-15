

module {
  llvm.func @lua_bool_and(%arg0: !llvm<"{ i32, i64 }">, %arg1: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.call @lua_convert_bool_like(%arg0) : (!llvm<"{ i32, i64 }">) -> !llvm.i1
    %1 = llvm.call @lua_convert_bool_like(%arg1) : (!llvm<"{ i32, i64 }">) -> !llvm.i1
    %2 = llvm.and %0, %1 : !llvm.i1
    %3 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %4 = llvm.alloca %3 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %5 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %6 = llvm.getelementptr %4[%5, %5] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %7 = llvm.mlir.constant(1 : i32) : !llvm.i32
    llvm.store %7, %6 : !llvm<"i32*">
    %8 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %9 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %10 = llvm.getelementptr %4[%8, %9] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %11 = llvm.zext %2 : !llvm.i1 to !llvm.i64
    llvm.store %11, %10 : !llvm<"i64*">
    %12 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %13 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %14 = llvm.getelementptr %4[%12, %13] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %15 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %16 = llvm.getelementptr %4[%15, %15] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %17 = llvm.load %14 : !llvm<"i64*">
    %18 = llvm.load %16 : !llvm<"i32*">
    %19 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %20 = llvm.insertvalue %18, %19[0] : !llvm<"{ i32, i64 }">
    %21 = llvm.insertvalue %17, %20[1] : !llvm<"{ i32, i64 }">
    llvm.return %21 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_convert_bool_like(%arg0: !llvm<"{ i32, i64 }">) -> !llvm.i1 {
    %0 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1 = llvm.mlir.constant(false) : !llvm.i1
    %2 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %3 = llvm.mlir.constant(true) : !llvm.i1
    %4 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %5 = llvm.alloca %4 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %6 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, i64 }">
    %7 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, i64 }">
    %8 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %9 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %10 = llvm.getelementptr %5[%8, %9] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %11 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %12 = llvm.getelementptr %5[%11, %11] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %6, %10 : !llvm<"i64*">
    llvm.store %7, %12 : !llvm<"i32*">
    %13 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %14 = llvm.getelementptr %5[%13, %13] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %15 = llvm.load %14 : !llvm<"i32*">
    %16 = llvm.icmp "eq" %15, %0 : !llvm.i32
    llvm.cond_br %16, ^bb1(%1 : !llvm.i1), ^bb2
  ^bb1(%17: !llvm.i1):  // 2 preds: ^bb0, ^bb5
    llvm.br ^bb6(%17 : !llvm.i1)
  ^bb2:  // pred: ^bb0
    %18 = llvm.icmp "eq" %15, %2 : !llvm.i32
    llvm.cond_br %18, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %19 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %20 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %21 = llvm.getelementptr %5[%19, %20] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %22 = llvm.load %21 : !llvm<"i64*">
    %23 = llvm.trunc %22 : !llvm.i64 to !llvm.i1
    llvm.br ^bb5(%23 : !llvm.i1)
  ^bb4:  // pred: ^bb2
    llvm.br ^bb5(%3 : !llvm.i1)
  ^bb5(%24: !llvm.i1):  // 2 preds: ^bb3, ^bb4
    llvm.br ^bb1(%24 : !llvm.i1)
  ^bb6(%25: !llvm.i1):  // pred: ^bb1
    llvm.br ^bb7
  ^bb7:  // pred: ^bb6
    llvm.return %25 : !llvm.i1
  }
}
