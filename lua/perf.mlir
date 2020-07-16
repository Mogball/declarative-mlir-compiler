

module {
  llvm.func @print_one(!llvm<"{ i32, i64 }">)
  llvm.func @luac_check_number_type(%arg0: !llvm<"{ i32, i64 }">, %arg1: !llvm<"{ i32, i64 }">) -> !llvm.i1 {
    %0 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %1 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %2 = llvm.alloca %1 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %3 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, i64 }">
    %4 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, i64 }">
    %5 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %6 = llvm.getelementptr %2[%5, %5] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %3, %6 : !llvm<"i32*">
    %7 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %8 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %9 = llvm.getelementptr %2[%7, %8] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %4, %9 : !llvm<"i64*">
    %10 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %11 = llvm.alloca %10 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %12 = llvm.extractvalue %arg1[0] : !llvm<"{ i32, i64 }">
    %13 = llvm.extractvalue %arg1[1] : !llvm<"{ i32, i64 }">
    %14 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %15 = llvm.getelementptr %11[%14, %14] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %12, %15 : !llvm<"i32*">
    %16 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %17 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %18 = llvm.getelementptr %11[%16, %17] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %13, %18 : !llvm<"i64*">
    %19 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %20 = llvm.getelementptr %2[%19, %19] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %21 = llvm.load %20 : !llvm<"i32*">
    %22 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %23 = llvm.getelementptr %11[%22, %22] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %24 = llvm.load %23 : !llvm<"i32*">
    %25 = llvm.icmp "eq" %21, %0 : !llvm.i32
    %26 = llvm.icmp "eq" %24, %0 : !llvm.i32
    %27 = llvm.and %25, %26 : !llvm.i1
    llvm.return %27 : !llvm.i1
  }
  llvm.func @lua_add(%arg0: !llvm<"{ i32, i64 }">, %arg1: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1 = llvm.alloca %0 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %2 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, i64 }">
    %3 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, i64 }">
    %4 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %5 = llvm.getelementptr %1[%4, %4] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %2, %5 : !llvm<"i32*">
    %6 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %7 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %8 = llvm.getelementptr %1[%6, %7] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %3, %8 : !llvm<"i64*">
    %9 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %10 = llvm.alloca %9 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %11 = llvm.extractvalue %arg1[0] : !llvm<"{ i32, i64 }">
    %12 = llvm.extractvalue %arg1[1] : !llvm<"{ i32, i64 }">
    %13 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %14 = llvm.getelementptr %10[%13, %13] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %11, %14 : !llvm<"i32*">
    %15 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %16 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %17 = llvm.getelementptr %10[%15, %16] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %12, %17 : !llvm<"i64*">
    %18 = llvm.call @luac_check_number_type(%arg0, %arg1) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm.i1
    llvm.cond_br %18, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %19 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %20 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %21 = llvm.getelementptr %1[%19, %20] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %22 = llvm.load %21 : !llvm<"i64*">
    %23 = llvm.bitcast %22 : !llvm.i64 to !llvm.double
    %24 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %25 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %26 = llvm.getelementptr %10[%24, %25] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %27 = llvm.load %26 : !llvm<"i64*">
    %28 = llvm.bitcast %27 : !llvm.i64 to !llvm.double
    %29 = llvm.fadd %23, %28 : !llvm.double
    %30 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %31 = llvm.alloca %30 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %32 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %33 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %34 = llvm.getelementptr %31[%33, %33] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %32, %34 : !llvm<"i32*">
    %35 = llvm.bitcast %29 : !llvm.double to !llvm.i64
    %36 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %37 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %38 = llvm.getelementptr %31[%36, %37] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %35, %38 : !llvm<"i64*">
    %39 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %40 = llvm.getelementptr %31[%39, %39] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %41 = llvm.load %40 : !llvm<"i32*">
    %42 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %43 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %44 = llvm.getelementptr %31[%42, %43] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %45 = llvm.load %44 : !llvm<"i64*">
    %46 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %47 = llvm.insertvalue %41, %46[0] : !llvm<"{ i32, i64 }">
    %48 = llvm.insertvalue %45, %47[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%48 : !llvm<"{ i32, i64 }">)
  ^bb2:  // pred: ^bb0
    %49 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %50 = llvm.alloca %49 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %51 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %52 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %53 = llvm.getelementptr %50[%52, %52] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %51, %53 : !llvm<"i32*">
    %54 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %55 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %56 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %57 = llvm.getelementptr %50[%55, %56] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %54, %57 : !llvm<"i64*">
    %58 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %59 = llvm.getelementptr %50[%58, %58] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %60 = llvm.load %59 : !llvm<"i32*">
    %61 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %62 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %63 = llvm.getelementptr %50[%61, %62] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %64 = llvm.load %63 : !llvm<"i64*">
    %65 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %66 = llvm.insertvalue %60, %65[0] : !llvm<"{ i32, i64 }">
    %67 = llvm.insertvalue %64, %66[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%67 : !llvm<"{ i32, i64 }">)
  ^bb3(%68: !llvm<"{ i32, i64 }">):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %68 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_sub(%arg0: !llvm<"{ i32, i64 }">, %arg1: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1 = llvm.alloca %0 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %2 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, i64 }">
    %3 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, i64 }">
    %4 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %5 = llvm.getelementptr %1[%4, %4] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %2, %5 : !llvm<"i32*">
    %6 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %7 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %8 = llvm.getelementptr %1[%6, %7] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %3, %8 : !llvm<"i64*">
    %9 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %10 = llvm.alloca %9 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %11 = llvm.extractvalue %arg1[0] : !llvm<"{ i32, i64 }">
    %12 = llvm.extractvalue %arg1[1] : !llvm<"{ i32, i64 }">
    %13 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %14 = llvm.getelementptr %10[%13, %13] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %11, %14 : !llvm<"i32*">
    %15 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %16 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %17 = llvm.getelementptr %10[%15, %16] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %12, %17 : !llvm<"i64*">
    %18 = llvm.call @luac_check_number_type(%arg0, %arg1) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm.i1
    llvm.cond_br %18, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %19 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %20 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %21 = llvm.getelementptr %1[%19, %20] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %22 = llvm.load %21 : !llvm<"i64*">
    %23 = llvm.bitcast %22 : !llvm.i64 to !llvm.double
    %24 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %25 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %26 = llvm.getelementptr %10[%24, %25] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %27 = llvm.load %26 : !llvm<"i64*">
    %28 = llvm.bitcast %27 : !llvm.i64 to !llvm.double
    %29 = llvm.fsub %23, %28 : !llvm.double
    %30 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %31 = llvm.alloca %30 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %32 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %33 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %34 = llvm.getelementptr %31[%33, %33] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %32, %34 : !llvm<"i32*">
    %35 = llvm.bitcast %29 : !llvm.double to !llvm.i64
    %36 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %37 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %38 = llvm.getelementptr %31[%36, %37] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %35, %38 : !llvm<"i64*">
    %39 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %40 = llvm.getelementptr %31[%39, %39] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %41 = llvm.load %40 : !llvm<"i32*">
    %42 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %43 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %44 = llvm.getelementptr %31[%42, %43] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %45 = llvm.load %44 : !llvm<"i64*">
    %46 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %47 = llvm.insertvalue %41, %46[0] : !llvm<"{ i32, i64 }">
    %48 = llvm.insertvalue %45, %47[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%48 : !llvm<"{ i32, i64 }">)
  ^bb2:  // pred: ^bb0
    %49 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %50 = llvm.alloca %49 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %51 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %52 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %53 = llvm.getelementptr %50[%52, %52] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %51, %53 : !llvm<"i32*">
    %54 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %55 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %56 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %57 = llvm.getelementptr %50[%55, %56] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %54, %57 : !llvm<"i64*">
    %58 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %59 = llvm.getelementptr %50[%58, %58] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %60 = llvm.load %59 : !llvm<"i32*">
    %61 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %62 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %63 = llvm.getelementptr %50[%61, %62] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %64 = llvm.load %63 : !llvm<"i64*">
    %65 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %66 = llvm.insertvalue %60, %65[0] : !llvm<"{ i32, i64 }">
    %67 = llvm.insertvalue %64, %66[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%67 : !llvm<"{ i32, i64 }">)
  ^bb3(%68: !llvm<"{ i32, i64 }">):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %68 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_mul(%arg0: !llvm<"{ i32, i64 }">, %arg1: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1 = llvm.alloca %0 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %2 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, i64 }">
    %3 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, i64 }">
    %4 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %5 = llvm.getelementptr %1[%4, %4] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %2, %5 : !llvm<"i32*">
    %6 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %7 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %8 = llvm.getelementptr %1[%6, %7] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %3, %8 : !llvm<"i64*">
    %9 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %10 = llvm.alloca %9 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %11 = llvm.extractvalue %arg1[0] : !llvm<"{ i32, i64 }">
    %12 = llvm.extractvalue %arg1[1] : !llvm<"{ i32, i64 }">
    %13 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %14 = llvm.getelementptr %10[%13, %13] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %11, %14 : !llvm<"i32*">
    %15 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %16 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %17 = llvm.getelementptr %10[%15, %16] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %12, %17 : !llvm<"i64*">
    %18 = llvm.call @luac_check_number_type(%arg0, %arg1) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm.i1
    llvm.cond_br %18, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %19 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %20 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %21 = llvm.getelementptr %1[%19, %20] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %22 = llvm.load %21 : !llvm<"i64*">
    %23 = llvm.bitcast %22 : !llvm.i64 to !llvm.double
    %24 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %25 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %26 = llvm.getelementptr %10[%24, %25] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %27 = llvm.load %26 : !llvm<"i64*">
    %28 = llvm.bitcast %27 : !llvm.i64 to !llvm.double
    %29 = llvm.fmul %23, %28 : !llvm.double
    %30 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %31 = llvm.alloca %30 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %32 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %33 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %34 = llvm.getelementptr %31[%33, %33] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %32, %34 : !llvm<"i32*">
    %35 = llvm.bitcast %29 : !llvm.double to !llvm.i64
    %36 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %37 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %38 = llvm.getelementptr %31[%36, %37] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %35, %38 : !llvm<"i64*">
    %39 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %40 = llvm.getelementptr %31[%39, %39] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %41 = llvm.load %40 : !llvm<"i32*">
    %42 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %43 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %44 = llvm.getelementptr %31[%42, %43] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %45 = llvm.load %44 : !llvm<"i64*">
    %46 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %47 = llvm.insertvalue %41, %46[0] : !llvm<"{ i32, i64 }">
    %48 = llvm.insertvalue %45, %47[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%48 : !llvm<"{ i32, i64 }">)
  ^bb2:  // pred: ^bb0
    %49 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %50 = llvm.alloca %49 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %51 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %52 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %53 = llvm.getelementptr %50[%52, %52] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %51, %53 : !llvm<"i32*">
    %54 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %55 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %56 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %57 = llvm.getelementptr %50[%55, %56] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %54, %57 : !llvm<"i64*">
    %58 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %59 = llvm.getelementptr %50[%58, %58] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %60 = llvm.load %59 : !llvm<"i32*">
    %61 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %62 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %63 = llvm.getelementptr %50[%61, %62] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %64 = llvm.load %63 : !llvm<"i64*">
    %65 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %66 = llvm.insertvalue %60, %65[0] : !llvm<"{ i32, i64 }">
    %67 = llvm.insertvalue %64, %66[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%67 : !llvm<"{ i32, i64 }">)
  ^bb3(%68: !llvm<"{ i32, i64 }">):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %68 : !llvm<"{ i32, i64 }">
  }
  llvm.func @pow(!llvm.double, !llvm.double) -> !llvm.double
  llvm.func @lua_pow(%arg0: !llvm<"{ i32, i64 }">, %arg1: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1 = llvm.alloca %0 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %2 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, i64 }">
    %3 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, i64 }">
    %4 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %5 = llvm.getelementptr %1[%4, %4] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %2, %5 : !llvm<"i32*">
    %6 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %7 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %8 = llvm.getelementptr %1[%6, %7] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %3, %8 : !llvm<"i64*">
    %9 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %10 = llvm.alloca %9 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %11 = llvm.extractvalue %arg1[0] : !llvm<"{ i32, i64 }">
    %12 = llvm.extractvalue %arg1[1] : !llvm<"{ i32, i64 }">
    %13 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %14 = llvm.getelementptr %10[%13, %13] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %11, %14 : !llvm<"i32*">
    %15 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %16 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %17 = llvm.getelementptr %10[%15, %16] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %12, %17 : !llvm<"i64*">
    %18 = llvm.call @luac_check_number_type(%arg0, %arg1) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm.i1
    llvm.cond_br %18, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %19 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %20 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %21 = llvm.getelementptr %1[%19, %20] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %22 = llvm.load %21 : !llvm<"i64*">
    %23 = llvm.bitcast %22 : !llvm.i64 to !llvm.double
    %24 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %25 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %26 = llvm.getelementptr %10[%24, %25] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %27 = llvm.load %26 : !llvm<"i64*">
    %28 = llvm.bitcast %27 : !llvm.i64 to !llvm.double
    %29 = llvm.call @pow(%23, %28) : (!llvm.double, !llvm.double) -> !llvm.double
    %30 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %31 = llvm.alloca %30 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %32 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %33 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %34 = llvm.getelementptr %31[%33, %33] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %32, %34 : !llvm<"i32*">
    %35 = llvm.bitcast %29 : !llvm.double to !llvm.i64
    %36 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %37 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %38 = llvm.getelementptr %31[%36, %37] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %35, %38 : !llvm<"i64*">
    %39 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %40 = llvm.getelementptr %31[%39, %39] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %41 = llvm.load %40 : !llvm<"i32*">
    %42 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %43 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %44 = llvm.getelementptr %31[%42, %43] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %45 = llvm.load %44 : !llvm<"i64*">
    %46 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %47 = llvm.insertvalue %41, %46[0] : !llvm<"{ i32, i64 }">
    %48 = llvm.insertvalue %45, %47[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%48 : !llvm<"{ i32, i64 }">)
  ^bb2:  // pred: ^bb0
    %49 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %50 = llvm.alloca %49 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %51 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %52 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %53 = llvm.getelementptr %50[%52, %52] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %51, %53 : !llvm<"i32*">
    %54 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %55 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %56 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %57 = llvm.getelementptr %50[%55, %56] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %54, %57 : !llvm<"i64*">
    %58 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %59 = llvm.getelementptr %50[%58, %58] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %60 = llvm.load %59 : !llvm<"i32*">
    %61 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %62 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %63 = llvm.getelementptr %50[%61, %62] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %64 = llvm.load %63 : !llvm<"i64*">
    %65 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %66 = llvm.insertvalue %60, %65[0] : !llvm<"{ i32, i64 }">
    %67 = llvm.insertvalue %64, %66[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%67 : !llvm<"{ i32, i64 }">)
  ^bb3(%68: !llvm<"{ i32, i64 }">):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %68 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_neg(%arg0: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %1 = llvm.mlir.constant(-1.000000e+00 : f64) : !llvm.double
    %2 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %3 = llvm.alloca %2 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %4 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, i64 }">
    %5 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, i64 }">
    %6 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %7 = llvm.getelementptr %3[%6, %6] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %4, %7 : !llvm<"i32*">
    %8 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %9 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %10 = llvm.getelementptr %3[%8, %9] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %5, %10 : !llvm<"i64*">
    %11 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %12 = llvm.getelementptr %3[%11, %11] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %13 = llvm.load %12 : !llvm<"i32*">
    %14 = llvm.icmp "eq" %0, %13 : !llvm.i32
    llvm.cond_br %14, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %15 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %16 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %17 = llvm.getelementptr %3[%15, %16] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %18 = llvm.load %17 : !llvm<"i64*">
    %19 = llvm.bitcast %18 : !llvm.i64 to !llvm.double
    %20 = llvm.fmul %1, %19 : !llvm.double
    %21 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %22 = llvm.alloca %21 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %23 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %24 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %25 = llvm.getelementptr %22[%24, %24] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %23, %25 : !llvm<"i32*">
    %26 = llvm.bitcast %20 : !llvm.double to !llvm.i64
    %27 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %28 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %29 = llvm.getelementptr %22[%27, %28] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %26, %29 : !llvm<"i64*">
    %30 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %31 = llvm.getelementptr %22[%30, %30] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %32 = llvm.load %31 : !llvm<"i32*">
    %33 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %34 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %35 = llvm.getelementptr %22[%33, %34] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %36 = llvm.load %35 : !llvm<"i64*">
    %37 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %38 = llvm.insertvalue %32, %37[0] : !llvm<"{ i32, i64 }">
    %39 = llvm.insertvalue %36, %38[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%39 : !llvm<"{ i32, i64 }">)
  ^bb2:  // pred: ^bb0
    %40 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %41 = llvm.alloca %40 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %42 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %43 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %44 = llvm.getelementptr %41[%43, %43] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %42, %44 : !llvm<"i32*">
    %45 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %46 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %47 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %48 = llvm.getelementptr %41[%46, %47] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %45, %48 : !llvm<"i64*">
    %49 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %50 = llvm.getelementptr %41[%49, %49] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %51 = llvm.load %50 : !llvm<"i32*">
    %52 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %53 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %54 = llvm.getelementptr %41[%52, %53] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %55 = llvm.load %54 : !llvm<"i64*">
    %56 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %57 = llvm.insertvalue %51, %56[0] : !llvm<"{ i32, i64 }">
    %58 = llvm.insertvalue %55, %57[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%58 : !llvm<"{ i32, i64 }">)
  ^bb3(%59: !llvm<"{ i32, i64 }">):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %59 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_lt(%arg0: !llvm<"{ i32, i64 }">, %arg1: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1 = llvm.alloca %0 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %2 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, i64 }">
    %3 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, i64 }">
    %4 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %5 = llvm.getelementptr %1[%4, %4] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %2, %5 : !llvm<"i32*">
    %6 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %7 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %8 = llvm.getelementptr %1[%6, %7] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %3, %8 : !llvm<"i64*">
    %9 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %10 = llvm.alloca %9 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %11 = llvm.extractvalue %arg1[0] : !llvm<"{ i32, i64 }">
    %12 = llvm.extractvalue %arg1[1] : !llvm<"{ i32, i64 }">
    %13 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %14 = llvm.getelementptr %10[%13, %13] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %11, %14 : !llvm<"i32*">
    %15 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %16 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %17 = llvm.getelementptr %10[%15, %16] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %12, %17 : !llvm<"i64*">
    %18 = llvm.call @luac_check_number_type(%arg0, %arg1) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm.i1
    llvm.cond_br %18, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %19 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %20 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %21 = llvm.getelementptr %1[%19, %20] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %22 = llvm.load %21 : !llvm<"i64*">
    %23 = llvm.bitcast %22 : !llvm.i64 to !llvm.double
    %24 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %25 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %26 = llvm.getelementptr %10[%24, %25] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %27 = llvm.load %26 : !llvm<"i64*">
    %28 = llvm.bitcast %27 : !llvm.i64 to !llvm.double
    %29 = llvm.fcmp "olt" %23, %28 : !llvm.double
    %30 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %31 = llvm.alloca %30 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %32 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %33 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %34 = llvm.getelementptr %31[%33, %33] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %32, %34 : !llvm<"i32*">
    %35 = llvm.zext %29 : !llvm.i1 to !llvm.i64
    %36 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %37 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %38 = llvm.getelementptr %31[%36, %37] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %35, %38 : !llvm<"i64*">
    %39 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %40 = llvm.getelementptr %31[%39, %39] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %41 = llvm.load %40 : !llvm<"i32*">
    %42 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %43 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %44 = llvm.getelementptr %31[%42, %43] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %45 = llvm.load %44 : !llvm<"i64*">
    %46 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %47 = llvm.insertvalue %41, %46[0] : !llvm<"{ i32, i64 }">
    %48 = llvm.insertvalue %45, %47[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%48 : !llvm<"{ i32, i64 }">)
  ^bb2:  // pred: ^bb0
    %49 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %50 = llvm.alloca %49 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %51 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %52 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %53 = llvm.getelementptr %50[%52, %52] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %51, %53 : !llvm<"i32*">
    %54 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %55 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %56 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %57 = llvm.getelementptr %50[%55, %56] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %54, %57 : !llvm<"i64*">
    %58 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %59 = llvm.getelementptr %50[%58, %58] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %60 = llvm.load %59 : !llvm<"i32*">
    %61 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %62 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %63 = llvm.getelementptr %50[%61, %62] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %64 = llvm.load %63 : !llvm<"i64*">
    %65 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %66 = llvm.insertvalue %60, %65[0] : !llvm<"{ i32, i64 }">
    %67 = llvm.insertvalue %64, %66[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%67 : !llvm<"{ i32, i64 }">)
  ^bb3(%68: !llvm<"{ i32, i64 }">):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %68 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_gt(%arg0: !llvm<"{ i32, i64 }">, %arg1: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1 = llvm.alloca %0 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %2 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, i64 }">
    %3 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, i64 }">
    %4 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %5 = llvm.getelementptr %1[%4, %4] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %2, %5 : !llvm<"i32*">
    %6 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %7 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %8 = llvm.getelementptr %1[%6, %7] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %3, %8 : !llvm<"i64*">
    %9 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %10 = llvm.alloca %9 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %11 = llvm.extractvalue %arg1[0] : !llvm<"{ i32, i64 }">
    %12 = llvm.extractvalue %arg1[1] : !llvm<"{ i32, i64 }">
    %13 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %14 = llvm.getelementptr %10[%13, %13] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %11, %14 : !llvm<"i32*">
    %15 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %16 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %17 = llvm.getelementptr %10[%15, %16] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %12, %17 : !llvm<"i64*">
    %18 = llvm.call @luac_check_number_type(%arg0, %arg1) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm.i1
    llvm.cond_br %18, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %19 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %20 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %21 = llvm.getelementptr %1[%19, %20] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %22 = llvm.load %21 : !llvm<"i64*">
    %23 = llvm.bitcast %22 : !llvm.i64 to !llvm.double
    %24 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %25 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %26 = llvm.getelementptr %10[%24, %25] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %27 = llvm.load %26 : !llvm<"i64*">
    %28 = llvm.bitcast %27 : !llvm.i64 to !llvm.double
    %29 = llvm.fcmp "ogt" %23, %28 : !llvm.double
    %30 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %31 = llvm.alloca %30 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %32 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %33 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %34 = llvm.getelementptr %31[%33, %33] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %32, %34 : !llvm<"i32*">
    %35 = llvm.zext %29 : !llvm.i1 to !llvm.i64
    %36 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %37 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %38 = llvm.getelementptr %31[%36, %37] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %35, %38 : !llvm<"i64*">
    %39 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %40 = llvm.getelementptr %31[%39, %39] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %41 = llvm.load %40 : !llvm<"i32*">
    %42 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %43 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %44 = llvm.getelementptr %31[%42, %43] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %45 = llvm.load %44 : !llvm<"i64*">
    %46 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %47 = llvm.insertvalue %41, %46[0] : !llvm<"{ i32, i64 }">
    %48 = llvm.insertvalue %45, %47[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%48 : !llvm<"{ i32, i64 }">)
  ^bb2:  // pred: ^bb0
    %49 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %50 = llvm.alloca %49 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %51 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %52 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %53 = llvm.getelementptr %50[%52, %52] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %51, %53 : !llvm<"i32*">
    %54 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %55 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %56 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %57 = llvm.getelementptr %50[%55, %56] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %54, %57 : !llvm<"i64*">
    %58 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %59 = llvm.getelementptr %50[%58, %58] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %60 = llvm.load %59 : !llvm<"i32*">
    %61 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %62 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %63 = llvm.getelementptr %50[%61, %62] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %64 = llvm.load %63 : !llvm<"i64*">
    %65 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %66 = llvm.insertvalue %60, %65[0] : !llvm<"{ i32, i64 }">
    %67 = llvm.insertvalue %64, %66[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%67 : !llvm<"{ i32, i64 }">)
  ^bb3(%68: !llvm<"{ i32, i64 }">):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %68 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_le(%arg0: !llvm<"{ i32, i64 }">, %arg1: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %1 = llvm.alloca %0 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %2 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, i64 }">
    %3 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, i64 }">
    %4 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %5 = llvm.getelementptr %1[%4, %4] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %2, %5 : !llvm<"i32*">
    %6 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %7 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %8 = llvm.getelementptr %1[%6, %7] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %3, %8 : !llvm<"i64*">
    %9 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %10 = llvm.alloca %9 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %11 = llvm.extractvalue %arg1[0] : !llvm<"{ i32, i64 }">
    %12 = llvm.extractvalue %arg1[1] : !llvm<"{ i32, i64 }">
    %13 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %14 = llvm.getelementptr %10[%13, %13] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %11, %14 : !llvm<"i32*">
    %15 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %16 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %17 = llvm.getelementptr %10[%15, %16] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %12, %17 : !llvm<"i64*">
    %18 = llvm.call @luac_check_number_type(%arg0, %arg1) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm.i1
    llvm.cond_br %18, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %19 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %20 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %21 = llvm.getelementptr %1[%19, %20] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %22 = llvm.load %21 : !llvm<"i64*">
    %23 = llvm.bitcast %22 : !llvm.i64 to !llvm.double
    %24 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %25 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %26 = llvm.getelementptr %10[%24, %25] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %27 = llvm.load %26 : !llvm<"i64*">
    %28 = llvm.bitcast %27 : !llvm.i64 to !llvm.double
    %29 = llvm.fcmp "ole" %23, %28 : !llvm.double
    %30 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %31 = llvm.alloca %30 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %32 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %33 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %34 = llvm.getelementptr %31[%33, %33] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %32, %34 : !llvm<"i32*">
    %35 = llvm.zext %29 : !llvm.i1 to !llvm.i64
    %36 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %37 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %38 = llvm.getelementptr %31[%36, %37] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %35, %38 : !llvm<"i64*">
    %39 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %40 = llvm.getelementptr %31[%39, %39] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %41 = llvm.load %40 : !llvm<"i32*">
    %42 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %43 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %44 = llvm.getelementptr %31[%42, %43] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %45 = llvm.load %44 : !llvm<"i64*">
    %46 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %47 = llvm.insertvalue %41, %46[0] : !llvm<"{ i32, i64 }">
    %48 = llvm.insertvalue %45, %47[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%48 : !llvm<"{ i32, i64 }">)
  ^bb2:  // pred: ^bb0
    %49 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %50 = llvm.alloca %49 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %51 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %52 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %53 = llvm.getelementptr %50[%52, %52] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %51, %53 : !llvm<"i32*">
    %54 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %55 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %56 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %57 = llvm.getelementptr %50[%55, %56] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %54, %57 : !llvm<"i64*">
    %58 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %59 = llvm.getelementptr %50[%58, %58] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %60 = llvm.load %59 : !llvm<"i32*">
    %61 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %62 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %63 = llvm.getelementptr %50[%61, %62] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %64 = llvm.load %63 : !llvm<"i64*">
    %65 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %66 = llvm.insertvalue %60, %65[0] : !llvm<"{ i32, i64 }">
    %67 = llvm.insertvalue %64, %66[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%67 : !llvm<"{ i32, i64 }">)
  ^bb3(%68: !llvm<"{ i32, i64 }">):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %68 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_bool_and(%arg0: !llvm<"{ i32, i64 }">, %arg1: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.call @lua_convert_bool_like(%arg0) : (!llvm<"{ i32, i64 }">) -> !llvm.i1
    %1 = llvm.call @lua_convert_bool_like(%arg1) : (!llvm<"{ i32, i64 }">) -> !llvm.i1
    %2 = llvm.and %0, %1 : !llvm.i1
    %3 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %4 = llvm.alloca %3 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %5 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %6 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %7 = llvm.getelementptr %4[%6, %6] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %5, %7 : !llvm<"i32*">
    %8 = llvm.zext %2 : !llvm.i1 to !llvm.i64
    %9 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %10 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %11 = llvm.getelementptr %4[%9, %10] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %8, %11 : !llvm<"i64*">
    %12 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %13 = llvm.getelementptr %4[%12, %12] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %14 = llvm.load %13 : !llvm<"i32*">
    %15 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %16 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %17 = llvm.getelementptr %4[%15, %16] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %18 = llvm.load %17 : !llvm<"i64*">
    %19 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %20 = llvm.insertvalue %14, %19[0] : !llvm<"{ i32, i64 }">
    %21 = llvm.insertvalue %18, %20[1] : !llvm<"{ i32, i64 }">
    llvm.return %21 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_bool_not(%arg0: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.mlir.constant(true) : !llvm.i1
    %1 = llvm.call @lua_convert_bool_like(%arg0) : (!llvm<"{ i32, i64 }">) -> !llvm.i1
    %2 = llvm.xor %1, %0 : !llvm.i1
    %3 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %4 = llvm.alloca %3 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %5 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %6 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %7 = llvm.getelementptr %4[%6, %6] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %5, %7 : !llvm<"i32*">
    %8 = llvm.zext %2 : !llvm.i1 to !llvm.i64
    %9 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %10 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %11 = llvm.getelementptr %4[%9, %10] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %8, %11 : !llvm<"i64*">
    %12 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %13 = llvm.getelementptr %4[%12, %12] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %14 = llvm.load %13 : !llvm<"i32*">
    %15 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %16 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %17 = llvm.getelementptr %4[%15, %16] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %18 = llvm.load %17 : !llvm<"i64*">
    %19 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %20 = llvm.insertvalue %14, %19[0] : !llvm<"{ i32, i64 }">
    %21 = llvm.insertvalue %18, %20[1] : !llvm<"{ i32, i64 }">
    llvm.return %21 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_list_size_impl(!llvm<"i8*">) -> !llvm.i64
  llvm.func @lua_list_size(%arg0: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.mlir.constant(4 : i32) : !llvm.i32
    %1 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %2 = llvm.alloca %1 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %3 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, i64 }">
    %4 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, i64 }">
    %5 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %6 = llvm.getelementptr %2[%5, %5] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %3, %6 : !llvm<"i32*">
    %7 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %8 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %9 = llvm.getelementptr %2[%7, %8] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %4, %9 : !llvm<"i64*">
    %10 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %11 = llvm.getelementptr %2[%10, %10] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %12 = llvm.load %11 : !llvm<"i32*">
    %13 = llvm.icmp "eq" %12, %0 : !llvm.i32
    llvm.cond_br %13, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %14 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %15 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %16 = llvm.getelementptr %2[%14, %15] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %17 = llvm.bitcast %16 : !llvm<"i64*"> to !llvm<"i8**">
    %18 = llvm.load %17 : !llvm<"i8**">
    %19 = llvm.call @lua_list_size_impl(%18) : (!llvm<"i8*">) -> !llvm.i64
    %20 = llvm.sitofp %19 : !llvm.i64 to !llvm.double
    %21 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %22 = llvm.alloca %21 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %23 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %24 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %25 = llvm.getelementptr %22[%24, %24] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %23, %25 : !llvm<"i32*">
    %26 = llvm.bitcast %20 : !llvm.double to !llvm.i64
    %27 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %28 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %29 = llvm.getelementptr %22[%27, %28] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %26, %29 : !llvm<"i64*">
    %30 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %31 = llvm.getelementptr %22[%30, %30] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %32 = llvm.load %31 : !llvm<"i32*">
    %33 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %34 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %35 = llvm.getelementptr %22[%33, %34] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %36 = llvm.load %35 : !llvm<"i64*">
    %37 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %38 = llvm.insertvalue %32, %37[0] : !llvm<"{ i32, i64 }">
    %39 = llvm.insertvalue %36, %38[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%39 : !llvm<"{ i32, i64 }">)
  ^bb2:  // pred: ^bb0
    %40 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %41 = llvm.alloca %40 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %42 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %43 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %44 = llvm.getelementptr %41[%43, %43] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %42, %44 : !llvm<"i32*">
    %45 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %46 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %47 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %48 = llvm.getelementptr %41[%46, %47] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %45, %48 : !llvm<"i64*">
    %49 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %50 = llvm.getelementptr %41[%49, %49] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %51 = llvm.load %50 : !llvm<"i32*">
    %52 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %53 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %54 = llvm.getelementptr %41[%52, %53] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %55 = llvm.load %54 : !llvm<"i64*">
    %56 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %57 = llvm.insertvalue %51, %56[0] : !llvm<"{ i32, i64 }">
    %58 = llvm.insertvalue %55, %57[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%58 : !llvm<"{ i32, i64 }">)
  ^bb3(%59: !llvm<"{ i32, i64 }">):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %59 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_strcat_impl(!llvm<"i8*">, !llvm<"i8*">) -> !llvm<"{ i32, i64 }">
  llvm.func @lua_strcat(%arg0: !llvm<"{ i32, i64 }">, %arg1: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.mlir.constant(3 : i32) : !llvm.i32
    %1 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %2 = llvm.alloca %1 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %3 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, i64 }">
    %4 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, i64 }">
    %5 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %6 = llvm.getelementptr %2[%5, %5] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %3, %6 : !llvm<"i32*">
    %7 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %8 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %9 = llvm.getelementptr %2[%7, %8] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %4, %9 : !llvm<"i64*">
    %10 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %11 = llvm.alloca %10 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %12 = llvm.extractvalue %arg1[0] : !llvm<"{ i32, i64 }">
    %13 = llvm.extractvalue %arg1[1] : !llvm<"{ i32, i64 }">
    %14 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %15 = llvm.getelementptr %11[%14, %14] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %12, %15 : !llvm<"i32*">
    %16 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %17 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %18 = llvm.getelementptr %11[%16, %17] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %13, %18 : !llvm<"i64*">
    %19 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %20 = llvm.getelementptr %2[%19, %19] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %21 = llvm.load %20 : !llvm<"i32*">
    %22 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %23 = llvm.getelementptr %11[%22, %22] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %24 = llvm.load %23 : !llvm<"i32*">
    %25 = llvm.icmp "eq" %21, %0 : !llvm.i32
    %26 = llvm.icmp "eq" %24, %0 : !llvm.i32
    %27 = llvm.and %25, %26 : !llvm.i1
    llvm.cond_br %27, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %28 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %29 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %30 = llvm.getelementptr %2[%28, %29] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %31 = llvm.bitcast %30 : !llvm<"i64*"> to !llvm<"i8**">
    %32 = llvm.load %31 : !llvm<"i8**">
    %33 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %34 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %35 = llvm.getelementptr %11[%33, %34] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %36 = llvm.bitcast %35 : !llvm<"i64*"> to !llvm<"i8**">
    %37 = llvm.load %36 : !llvm<"i8**">
    %38 = llvm.call @lua_strcat_impl(%32, %37) : (!llvm<"i8*">, !llvm<"i8*">) -> !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%38 : !llvm<"{ i32, i64 }">)
  ^bb2:  // pred: ^bb0
    %39 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %40 = llvm.alloca %39 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %41 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %42 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %43 = llvm.getelementptr %40[%42, %42] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %41, %43 : !llvm<"i32*">
    %44 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %45 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %46 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %47 = llvm.getelementptr %40[%45, %46] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %44, %47 : !llvm<"i64*">
    %48 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %49 = llvm.getelementptr %40[%48, %48] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %50 = llvm.load %49 : !llvm<"i32*">
    %51 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %52 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %53 = llvm.getelementptr %40[%51, %52] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %54 = llvm.load %53 : !llvm<"i64*">
    %55 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %56 = llvm.insertvalue %50, %55[0] : !llvm<"{ i32, i64 }">
    %57 = llvm.insertvalue %54, %56[1] : !llvm<"{ i32, i64 }">
    llvm.br ^bb3(%57 : !llvm<"{ i32, i64 }">)
  ^bb3(%58: !llvm<"{ i32, i64 }">):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %58 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_eq_impl(!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm.i1
  llvm.func @lua_eq(%arg0: !llvm<"{ i32, i64 }">, %arg1: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.mlir.constant(false) : !llvm.i1
    %1 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %2 = llvm.alloca %1 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %3 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, i64 }">
    %4 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, i64 }">
    %5 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %6 = llvm.getelementptr %2[%5, %5] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %3, %6 : !llvm<"i32*">
    %7 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %8 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %9 = llvm.getelementptr %2[%7, %8] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %4, %9 : !llvm<"i64*">
    %10 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %11 = llvm.alloca %10 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %12 = llvm.extractvalue %arg1[0] : !llvm<"{ i32, i64 }">
    %13 = llvm.extractvalue %arg1[1] : !llvm<"{ i32, i64 }">
    %14 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %15 = llvm.getelementptr %11[%14, %14] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %12, %15 : !llvm<"i32*">
    %16 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %17 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %18 = llvm.getelementptr %11[%16, %17] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %13, %18 : !llvm<"i64*">
    %19 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %20 = llvm.getelementptr %2[%19, %19] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %21 = llvm.load %20 : !llvm<"i32*">
    %22 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %23 = llvm.getelementptr %11[%22, %22] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %24 = llvm.load %23 : !llvm<"i32*">
    %25 = llvm.icmp "eq" %21, %24 : !llvm.i32
    llvm.cond_br %25, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %26 = llvm.call @lua_eq_impl(%arg0, %arg1) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm.i1
    llvm.br ^bb3(%26 : !llvm.i1)
  ^bb2:  // pred: ^bb0
    llvm.br ^bb3(%0 : !llvm.i1)
  ^bb3(%27: !llvm.i1):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    %28 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %29 = llvm.alloca %28 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %30 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %31 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %32 = llvm.getelementptr %29[%31, %31] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %30, %32 : !llvm<"i32*">
    %33 = llvm.zext %27 : !llvm.i1 to !llvm.i64
    %34 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %35 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %36 = llvm.getelementptr %29[%34, %35] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %33, %36 : !llvm<"i64*">
    %37 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %38 = llvm.getelementptr %29[%37, %37] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %39 = llvm.load %38 : !llvm<"i32*">
    %40 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %41 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %42 = llvm.getelementptr %29[%40, %41] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %43 = llvm.load %42 : !llvm<"i64*">
    %44 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %45 = llvm.insertvalue %39, %44[0] : !llvm<"{ i32, i64 }">
    %46 = llvm.insertvalue %43, %45[1] : !llvm<"{ i32, i64 }">
    llvm.return %46 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_ne(%arg0: !llvm<"{ i32, i64 }">, %arg1: !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }"> {
    %0 = llvm.mlir.constant(true) : !llvm.i1
    %1 = llvm.call @lua_eq(%arg0, %arg1) : (!llvm<"{ i32, i64 }">, !llvm<"{ i32, i64 }">) -> !llvm<"{ i32, i64 }">
    %2 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %3 = llvm.alloca %2 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %4 = llvm.extractvalue %1[0] : !llvm<"{ i32, i64 }">
    %5 = llvm.extractvalue %1[1] : !llvm<"{ i32, i64 }">
    %6 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %7 = llvm.getelementptr %3[%6, %6] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %4, %7 : !llvm<"i32*">
    %8 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %9 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %10 = llvm.getelementptr %3[%8, %9] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %5, %10 : !llvm<"i64*">
    %11 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %12 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %13 = llvm.getelementptr %3[%11, %12] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %14 = llvm.load %13 : !llvm<"i64*">
    %15 = llvm.trunc %14 : !llvm.i64 to !llvm.i1
    %16 = llvm.xor %15, %0 : !llvm.i1
    %17 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %18 = llvm.alloca %17 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %19 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %20 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %21 = llvm.getelementptr %18[%20, %20] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %19, %21 : !llvm<"i32*">
    %22 = llvm.zext %16 : !llvm.i1 to !llvm.i64
    %23 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %24 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %25 = llvm.getelementptr %18[%23, %24] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %22, %25 : !llvm<"i64*">
    %26 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %27 = llvm.getelementptr %18[%26, %26] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %28 = llvm.load %27 : !llvm<"i32*">
    %29 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %30 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %31 = llvm.getelementptr %18[%29, %30] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %32 = llvm.load %31 : !llvm<"i64*">
    %33 = llvm.mlir.undef : !llvm<"{ i32, i64 }">
    %34 = llvm.insertvalue %28, %33[0] : !llvm<"{ i32, i64 }">
    %35 = llvm.insertvalue %32, %34[1] : !llvm<"{ i32, i64 }">
    llvm.return %35 : !llvm<"{ i32, i64 }">
  }
  llvm.func @lua_convert_bool_like(%arg0: !llvm<"{ i32, i64 }">) -> !llvm.i1 {
    %0 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1 = llvm.mlir.constant(false) : !llvm.i1
    %2 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %3 = llvm.mlir.constant(true) : !llvm.i1
    %4 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %5 = llvm.alloca %4 x !llvm<"{ i32, i64 }"> {alignment = 8 : i64} : (!llvm.i32) -> !llvm<"{ i32, i64 }*">
    %6 = llvm.extractvalue %arg0[0] : !llvm<"{ i32, i64 }">
    %7 = llvm.extractvalue %arg0[1] : !llvm<"{ i32, i64 }">
    %8 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %9 = llvm.getelementptr %5[%8, %8] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    llvm.store %6, %9 : !llvm<"i32*">
    %10 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %11 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %12 = llvm.getelementptr %5[%10, %11] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %7, %12 : !llvm<"i64*">
    %13 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %14 = llvm.getelementptr %5[%13, %13] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i32*">
    %15 = llvm.load %14 : !llvm<"i32*">
    %16 = llvm.icmp "eq" %15, %0 : !llvm.i32
    llvm.cond_br %16, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.br ^bb7(%1 : !llvm.i1)
  ^bb2:  // pred: ^bb0
    %17 = llvm.icmp "eq" %15, %2 : !llvm.i32
    llvm.cond_br %17, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %18 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %19 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %20 = llvm.getelementptr %5[%18, %19] : (!llvm<"{ i32, i64 }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %21 = llvm.load %20 : !llvm<"i64*">
    %22 = llvm.trunc %21 : !llvm.i64 to !llvm.i1
    llvm.br ^bb5(%22 : !llvm.i1)
  ^bb4:  // pred: ^bb2
    llvm.br ^bb5(%3 : !llvm.i1)
  ^bb5(%23: !llvm.i1):  // 2 preds: ^bb3, ^bb4
    llvm.br ^bb6
  ^bb6:  // pred: ^bb5
    llvm.br ^bb7(%23 : !llvm.i1)
  ^bb7(%24: !llvm.i1):  // 2 preds: ^bb1, ^bb6
    llvm.br ^bb8
  ^bb8:  // pred: ^bb7
    llvm.return %24 : !llvm.i1
  }
}
