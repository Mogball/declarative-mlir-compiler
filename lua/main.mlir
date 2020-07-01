

module {
  llvm.func @main() {
    %0 = llvm.mlir.constant(5 : i64) : !llvm.i64
    %1 = llvm.call @lua_alloc() : () -> !llvm<"{ i16, i16, { i64 } }*">
    %2 = llvm.mlir.constant(2 : i16) : !llvm.i16
    llvm.call @lua_set_int64_val(%1, %0) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i64) -> ()
    llvm.call @lua_set_type(%1, %2) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i16) -> ()
    %3 = llvm.mlir.constant(6 : i64) : !llvm.i64
    %4 = llvm.call @lua_alloc() : () -> !llvm<"{ i16, i16, { i64 } }*">
    llvm.call @lua_set_int64_val(%4, %3) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i64) -> ()
    llvm.call @lua_set_type(%4, %2) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i16) -> ()
    %5 = llvm.mlir.constant(8 : i64) : !llvm.i64
    %6 = llvm.call @lua_alloc() : () -> !llvm<"{ i16, i16, { i64 } }*">
    llvm.call @lua_set_int64_val(%6, %5) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i64) -> ()
    llvm.call @lua_set_type(%6, %2) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i16) -> ()
    %7 = llvm.mlir.constant(9 : i64) : !llvm.i64
    %8 = llvm.call @lua_alloc() : () -> !llvm<"{ i16, i16, { i64 } }*">
    llvm.call @lua_set_int64_val(%8, %7) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i64) -> ()
    llvm.call @lua_set_type(%8, %2) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i16) -> ()
    %9 = llvm.call @lua_alloc() : () -> !llvm<"{ i16, i16, { i64 } }*">
    %10 = llvm.mlir.constant(0 : i16) : !llvm.i16
    llvm.call @lua_set_type(%9, %10) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i16) -> ()
    %11 = llvm.call @lua_builtin_print() : () -> !llvm<"{ i16, i16, { i64 } }*">
    %12 = llvm.call @lua_alloc() : () -> !llvm<"{ i16, i16, { i64 } }*">
    llvm.call @lua_set_int64_val(%12, %0) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i64) -> ()
    llvm.call @lua_set_type(%12, %2) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i16) -> ()
    %13 = llvm.call @lua_alloc() : () -> !llvm<"{ i16, i16, { i64 } }*">
    llvm.call @lua_set_int64_val(%13, %3) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i64) -> ()
    llvm.call @lua_set_type(%13, %2) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i16) -> ()
    %14 = llvm.mlir.constant(2 : i64) : !llvm.i64
    %15 = llvm.call @lua_new_pack(%14) : (!llvm.i64) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">
    llvm.call @lua_pack_push(%15, %12) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm<"{ i16, i16, { i64 } }*">) -> ()
    llvm.call @lua_pack_push(%15, %13) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm<"{ i16, i16, { i64 } }*">) -> ()
    %16 = llvm.call @lua_get_fcn_addr(%11) : (!llvm<"{ i16, i16, { i64 } }*">) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }* ({ i64, i64, { i16, i16, { i64 } }** }*)*">
    %17 = llvm.call %16(%15) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">
    llvm.call @lua_delete_pack(%17) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) -> ()
    %18 = llvm.call @lua_new_pack(%14) : (!llvm.i64) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">
    llvm.call @lua_pack_push(%18, %1) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm<"{ i16, i16, { i64 } }*">) -> ()
    llvm.call @lua_pack_push(%18, %4) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm<"{ i16, i16, { i64 } }*">) -> ()
    %19 = llvm.call @lua_get_fcn_addr(%11) : (!llvm<"{ i16, i16, { i64 } }*">) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }* ({ i64, i64, { i16, i16, { i64 } }** }*)*">
    %20 = llvm.call %19(%18) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">
    llvm.call @lua_delete_pack(%20) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) -> ()
    %21 = llvm.mlir.constant(3 : i64) : !llvm.i64
    %22 = llvm.call @lua_new_pack(%21) : (!llvm.i64) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">
    llvm.call @lua_pack_push(%22, %6) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm<"{ i16, i16, { i64 } }*">) -> ()
    llvm.call @lua_pack_push(%22, %8) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm<"{ i16, i16, { i64 } }*">) -> ()
    llvm.call @lua_pack_push(%22, %9) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm<"{ i16, i16, { i64 } }*">) -> ()
    %23 = llvm.call @lua_get_fcn_addr(%11) : (!llvm<"{ i16, i16, { i64 } }*">) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }* ({ i64, i64, { i16, i16, { i64 } }** }*)*">
    %24 = llvm.call %23(%22) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">
    llvm.call @lua_delete_pack(%24) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) -> ()
    %25 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %26 = llvm.call @lua_alloc() : () -> !llvm<"{ i16, i16, { i64 } }*">
    llvm.call @lua_set_int64_val(%26, %25) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i64) -> ()
    llvm.call @lua_set_type(%26, %2) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i16) -> ()
    %27 = llvm.call @lua_add(%1, %4) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm<"{ i16, i16, { i64 } }*">) -> !llvm<"{ i16, i16, { i64 } }*">
    %28 = llvm.call @lua_add(%27, %6) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm<"{ i16, i16, { i64 } }*">) -> !llvm<"{ i16, i16, { i64 } }*">
    %29 = llvm.call @lua_add(%28, %8) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm<"{ i16, i16, { i64 } }*">) -> !llvm<"{ i16, i16, { i64 } }*">
    %30 = llvm.call @lua_add(%29, %26) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm<"{ i16, i16, { i64 } }*">) -> !llvm<"{ i16, i16, { i64 } }*">
    %31 = llvm.call @lua_mul(%4, %6) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm<"{ i16, i16, { i64 } }*">) -> !llvm<"{ i16, i16, { i64 } }*">
    %32 = llvm.call @lua_add(%1, %31) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm<"{ i16, i16, { i64 } }*">) -> !llvm<"{ i16, i16, { i64 } }*">
    %33 = llvm.call @lua_add(%1, %4) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm<"{ i16, i16, { i64 } }*">) -> !llvm<"{ i16, i16, { i64 } }*">
    %34 = llvm.call @lua_mul(%33, %6) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm<"{ i16, i16, { i64 } }*">) -> !llvm<"{ i16, i16, { i64 } }*">
    %35 = llvm.call @lua_new_pack(%21) : (!llvm.i64) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">
    llvm.call @lua_pack_push(%35, %30) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm<"{ i16, i16, { i64 } }*">) -> ()
    llvm.call @lua_pack_push(%35, %32) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm<"{ i16, i16, { i64 } }*">) -> ()
    llvm.call @lua_pack_push(%35, %34) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm<"{ i16, i16, { i64 } }*">) -> ()
    %36 = llvm.call @lua_get_fcn_addr(%11) : (!llvm<"{ i16, i16, { i64 } }*">) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }* ({ i64, i64, { i16, i16, { i64 } }** }*)*">
    %37 = llvm.call %36(%35) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">
    llvm.call @lua_delete_pack(%37) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) -> ()
    %38 = llvm.mlir.constant(1 : i64) : !llvm.i64
    %39 = llvm.call @lua_new_pack(%38) : (!llvm.i64) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">
    llvm.call @lua_pack_push(%39, %4) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm<"{ i16, i16, { i64 } }*">) -> ()
    %40 = llvm.call @lua_get_fcn_addr(%11) : (!llvm<"{ i16, i16, { i64 } }*">) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }* ({ i64, i64, { i16, i16, { i64 } }** }*)*">
    %41 = llvm.call %40(%39) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">
    %42 = llvm.call @lua_pack_get_size(%41) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) -> !llvm.i64
    %43 = llvm.add %14, %42 : !llvm.i64
    %44 = llvm.call @lua_new_pack(%43) : (!llvm.i64) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">
    llvm.call @lua_pack_push(%44, %1) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm<"{ i16, i16, { i64 } }*">) -> ()
    llvm.call @lua_pack_push_all(%44, %41) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) -> ()
    llvm.call @lua_delete_pack(%41) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) -> ()
    llvm.call @lua_pack_push(%44, %6) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm<"{ i16, i16, { i64 } }*">) -> ()
    %45 = llvm.call @lua_get_fcn_addr(%11) : (!llvm<"{ i16, i16, { i64 } }*">) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }* ({ i64, i64, { i16, i16, { i64 } }** }*)*">
    %46 = llvm.call %45(%44) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">
    llvm.call @lua_delete_pack(%46) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) -> ()
    %47 = llvm.call @lua_alloc() : () -> !llvm<"{ i16, i16, { i64 } }*">
    llvm.call @lua_set_int64_val(%47, %0) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i64) -> ()
    llvm.call @lua_set_type(%47, %2) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i16) -> ()
    %48 = llvm.call @lua_alloc() : () -> !llvm<"{ i16, i16, { i64 } }*">
    llvm.call @lua_set_int64_val(%48, %3) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i64) -> ()
    llvm.call @lua_set_type(%48, %2) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i16) -> ()
    %49 = llvm.call @lua_alloc() : () -> !llvm<"{ i16, i16, { i64 } }*">
    llvm.call @lua_set_int64_val(%49, %5) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i64) -> ()
    llvm.call @lua_set_type(%49, %2) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i16) -> ()
    %50 = llvm.call @lua_new_pack(%38) : (!llvm.i64) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">
    llvm.call @lua_pack_push(%50, %48) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm<"{ i16, i16, { i64 } }*">) -> ()
    %51 = llvm.call @lua_get_fcn_addr(%11) : (!llvm<"{ i16, i16, { i64 } }*">) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }* ({ i64, i64, { i16, i16, { i64 } }** }*)*">
    %52 = llvm.call %51(%50) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">
    %53 = llvm.call @lua_pack_get_size(%52) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) -> !llvm.i64
    %54 = llvm.add %14, %53 : !llvm.i64
    %55 = llvm.call @lua_new_pack(%54) : (!llvm.i64) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">
    llvm.call @lua_pack_push(%55, %47) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm<"{ i16, i16, { i64 } }*">) -> ()
    llvm.call @lua_pack_push_all(%55, %52) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) -> ()
    llvm.call @lua_delete_pack(%52) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) -> ()
    llvm.call @lua_pack_push(%55, %49) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm<"{ i16, i16, { i64 } }*">) -> ()
    %56 = llvm.call @lua_pack_pull_one(%55) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) -> !llvm<"{ i16, i16, { i64 } }*">
    %57 = llvm.call @lua_pack_pull_one(%55) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) -> !llvm<"{ i16, i16, { i64 } }*">
    %58 = llvm.call @lua_pack_pull_one(%55) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) -> !llvm<"{ i16, i16, { i64 } }*">
    %59 = llvm.call @lua_pack_pull_one(%55) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) -> !llvm<"{ i16, i16, { i64 } }*">
    llvm.call @lua_delete_pack(%55) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) -> ()
    %60 = llvm.mlir.constant(4 : i64) : !llvm.i64
    %61 = llvm.call @lua_new_pack(%60) : (!llvm.i64) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">
    llvm.call @lua_pack_push(%61, %56) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm<"{ i16, i16, { i64 } }*">) -> ()
    llvm.call @lua_pack_push(%61, %57) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm<"{ i16, i16, { i64 } }*">) -> ()
    llvm.call @lua_pack_push(%61, %58) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm<"{ i16, i16, { i64 } }*">) -> ()
    llvm.call @lua_pack_push(%61, %59) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm<"{ i16, i16, { i64 } }*">) -> ()
    %62 = llvm.call @lua_get_fcn_addr(%11) : (!llvm<"{ i16, i16, { i64 } }*">) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }* ({ i64, i64, { i16, i16, { i64 } }** }*)*">
    %63 = llvm.call %62(%61) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">
    llvm.call @lua_delete_pack(%63) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) -> ()
    llvm.return
  }
  llvm.func @luac_check_number_type(%arg0: !llvm<"{ i16, i16, { i64 } }*">, %arg1: !llvm<"{ i16, i16, { i64 } }*">) -> !llvm<"{ i1, i16 }"> {
    %0 = llvm.mlir.constant(2 : i16) : !llvm.i16
    %1 = llvm.call @lua_get_type(%arg0) : (!llvm<"{ i16, i16, { i64 } }*">) -> !llvm.i16
    %2 = llvm.call @lua_get_type(%arg1) : (!llvm<"{ i16, i16, { i64 } }*">) -> !llvm.i16
    %3 = llvm.icmp "eq" %1, %0 : !llvm.i16
    %4 = llvm.icmp "eq" %2, %0 : !llvm.i16
    %5 = llvm.and %3, %4 : !llvm.i1
    %6 = llvm.mlir.undef : !llvm<"{ i1, i16 }">
    %7 = llvm.insertvalue %5, %6[0] : !llvm<"{ i1, i16 }">
    %8 = llvm.insertvalue %0, %7[1] : !llvm<"{ i1, i16 }">
    llvm.return %8 : !llvm<"{ i1, i16 }">
  }
  llvm.func @luac_get_as_fp(%arg0: !llvm<"{ i16, i16, { i64 } }*">, %arg1: !llvm.i1) -> !llvm.double {
    llvm.cond_br %arg1, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %0 = llvm.call @lua_get_int64_val(%arg0) : (!llvm<"{ i16, i16, { i64 } }*">) -> !llvm.i64
    %1 = llvm.sitofp %0 : !llvm.i64 to !llvm.double
    llvm.br ^bb3(%1 : !llvm.double)
  ^bb2:  // pred: ^bb0
    %2 = llvm.call @lua_get_double_val(%arg0) : (!llvm<"{ i16, i16, { i64 } }*">) -> !llvm.double
    llvm.br ^bb3(%2 : !llvm.double)
  ^bb3(%3: !llvm.double):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    llvm.return %3 : !llvm.double
  }
  llvm.func @lua_add(%arg0: !llvm<"{ i16, i16, { i64 } }*">, %arg1: !llvm<"{ i16, i16, { i64 } }*">) -> !llvm<"{ i16, i16, { i64 } }*"> {
    %0 = llvm.call @lua_alloc() : () -> !llvm<"{ i16, i16, { i64 } }*">
    %1 = llvm.mlir.constant(0 : i16) : !llvm.i16
    llvm.call @lua_set_type(%0, %1) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i16) -> ()
    %2 = llvm.call @luac_check_number_type(%arg0, %arg1) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm<"{ i16, i16, { i64 } }*">) -> !llvm<"{ i1, i16 }">
    %3 = llvm.extractvalue %2[0] : !llvm<"{ i1, i16 }">
    %4 = llvm.extractvalue %2[1] : !llvm<"{ i1, i16 }">
    llvm.cond_br %3, ^bb1, ^bb5
  ^bb1:  // pred: ^bb0
    llvm.call @lua_set_type(%0, %4) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i16) -> ()
    %5 = llvm.call @lua_is_int(%arg0) : (!llvm<"{ i16, i16, { i64 } }*">) -> !llvm.i1
    %6 = llvm.call @lua_is_int(%arg1) : (!llvm<"{ i16, i16, { i64 } }*">) -> !llvm.i1
    %7 = llvm.and %5, %6 : !llvm.i1
    llvm.cond_br %7, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %8 = llvm.call @lua_get_int64_val(%arg0) : (!llvm<"{ i16, i16, { i64 } }*">) -> !llvm.i64
    %9 = llvm.call @lua_get_int64_val(%arg1) : (!llvm<"{ i16, i16, { i64 } }*">) -> !llvm.i64
    %10 = llvm.add %8, %9 : !llvm.i64
    llvm.call @lua_set_int64_val(%0, %10) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i64) -> ()
    llvm.br ^bb4
  ^bb3:  // pred: ^bb1
    %11 = llvm.call @luac_get_as_fp(%arg0, %5) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i1) -> !llvm.double
    %12 = llvm.call @luac_get_as_fp(%arg1, %6) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i1) -> !llvm.double
    %13 = llvm.fadd %11, %12 : !llvm.double
    llvm.call @lua_set_double_val(%0, %13) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.double) -> ()
    llvm.br ^bb4
  ^bb4:  // 2 preds: ^bb2, ^bb3
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb0, ^bb4
    llvm.return %0 : !llvm<"{ i16, i16, { i64 } }*">
  }
  llvm.func @lua_sub(%arg0: !llvm<"{ i16, i16, { i64 } }*">, %arg1: !llvm<"{ i16, i16, { i64 } }*">) -> !llvm<"{ i16, i16, { i64 } }*"> {
    %0 = llvm.call @lua_alloc() : () -> !llvm<"{ i16, i16, { i64 } }*">
    %1 = llvm.mlir.constant(0 : i16) : !llvm.i16
    llvm.call @lua_set_type(%0, %1) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i16) -> ()
    %2 = llvm.call @luac_check_number_type(%arg0, %arg1) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm<"{ i16, i16, { i64 } }*">) -> !llvm<"{ i1, i16 }">
    %3 = llvm.extractvalue %2[0] : !llvm<"{ i1, i16 }">
    %4 = llvm.extractvalue %2[1] : !llvm<"{ i1, i16 }">
    llvm.cond_br %3, ^bb1, ^bb5
  ^bb1:  // pred: ^bb0
    llvm.call @lua_set_type(%0, %4) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i16) -> ()
    %5 = llvm.call @lua_is_int(%arg0) : (!llvm<"{ i16, i16, { i64 } }*">) -> !llvm.i1
    %6 = llvm.call @lua_is_int(%arg1) : (!llvm<"{ i16, i16, { i64 } }*">) -> !llvm.i1
    %7 = llvm.and %5, %6 : !llvm.i1
    llvm.cond_br %7, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %8 = llvm.call @lua_get_int64_val(%arg0) : (!llvm<"{ i16, i16, { i64 } }*">) -> !llvm.i64
    %9 = llvm.call @lua_get_int64_val(%arg1) : (!llvm<"{ i16, i16, { i64 } }*">) -> !llvm.i64
    %10 = llvm.sub %8, %9 : !llvm.i64
    llvm.call @lua_set_int64_val(%0, %10) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i64) -> ()
    llvm.br ^bb4
  ^bb3:  // pred: ^bb1
    %11 = llvm.call @luac_get_as_fp(%arg0, %5) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i1) -> !llvm.double
    %12 = llvm.call @luac_get_as_fp(%arg1, %6) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i1) -> !llvm.double
    %13 = llvm.fsub %11, %12 : !llvm.double
    llvm.call @lua_set_double_val(%0, %13) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.double) -> ()
    llvm.br ^bb4
  ^bb4:  // 2 preds: ^bb2, ^bb3
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb0, ^bb4
    llvm.return %0 : !llvm<"{ i16, i16, { i64 } }*">
  }
  llvm.func @lua_mul(%arg0: !llvm<"{ i16, i16, { i64 } }*">, %arg1: !llvm<"{ i16, i16, { i64 } }*">) -> !llvm<"{ i16, i16, { i64 } }*"> {
    %0 = llvm.call @lua_alloc() : () -> !llvm<"{ i16, i16, { i64 } }*">
    %1 = llvm.mlir.constant(0 : i16) : !llvm.i16
    llvm.call @lua_set_type(%0, %1) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i16) -> ()
    %2 = llvm.call @luac_check_number_type(%arg0, %arg1) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm<"{ i16, i16, { i64 } }*">) -> !llvm<"{ i1, i16 }">
    %3 = llvm.extractvalue %2[0] : !llvm<"{ i1, i16 }">
    %4 = llvm.extractvalue %2[1] : !llvm<"{ i1, i16 }">
    llvm.cond_br %3, ^bb1, ^bb5
  ^bb1:  // pred: ^bb0
    llvm.call @lua_set_type(%0, %4) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i16) -> ()
    %5 = llvm.call @lua_is_int(%arg0) : (!llvm<"{ i16, i16, { i64 } }*">) -> !llvm.i1
    %6 = llvm.call @lua_is_int(%arg1) : (!llvm<"{ i16, i16, { i64 } }*">) -> !llvm.i1
    %7 = llvm.and %5, %6 : !llvm.i1
    llvm.cond_br %7, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %8 = llvm.call @lua_get_int64_val(%arg0) : (!llvm<"{ i16, i16, { i64 } }*">) -> !llvm.i64
    %9 = llvm.call @lua_get_int64_val(%arg1) : (!llvm<"{ i16, i16, { i64 } }*">) -> !llvm.i64
    %10 = llvm.mul %8, %9 : !llvm.i64
    llvm.call @lua_set_int64_val(%0, %10) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i64) -> ()
    llvm.br ^bb4
  ^bb3:  // pred: ^bb1
    %11 = llvm.call @luac_get_as_fp(%arg0, %5) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i1) -> !llvm.double
    %12 = llvm.call @luac_get_as_fp(%arg1, %6) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i1) -> !llvm.double
    %13 = llvm.fmul %11, %12 : !llvm.double
    llvm.call @lua_set_double_val(%0, %13) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.double) -> ()
    llvm.br ^bb4
  ^bb4:  // 2 preds: ^bb2, ^bb3
    llvm.br ^bb5
  ^bb5:  // 2 preds: ^bb0, ^bb4
    llvm.return %0 : !llvm<"{ i16, i16, { i64 } }*">
  }
  llvm.mlir.global external @builtin_print() : !llvm<"{ i16, i16, { i64 } }*">
  llvm.func @lua_alloc() -> !llvm<"{ i16, i16, { i64 } }*"> {
    %0 = llvm.mlir.constant(16 : i64) : !llvm.i64
    %1 = llvm.call @malloc(%0) : (!llvm.i64) -> !llvm<"i8*">
    %2 = llvm.bitcast %1 : !llvm<"i8*"> to !llvm<"{ i16, i16, { i64 } }*">
    llvm.return %2 : !llvm<"{ i16, i16, { i64 } }*">
  }
  llvm.func @malloc(!llvm.i64) -> !llvm<"i8*">
  llvm.func @lua_get_type(%arg0: !llvm<"{ i16, i16, { i64 } }*">) -> !llvm.i16 {
    %0 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %2 = llvm.alloca %1 x !llvm<"{ i16, i16, { i64 } }*"> : (!llvm.i32) -> !llvm<"{ i16, i16, { i64 } }**">
    llvm.store %arg0, %2 : !llvm<"{ i16, i16, { i64 } }**">
    %3 = llvm.load %2 : !llvm<"{ i16, i16, { i64 } }**">
    %4 = llvm.getelementptr %3[%0, %0] : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i32, !llvm.i32) -> !llvm<"i16*">
    %5 = llvm.load %4 : !llvm<"i16*">
    llvm.return %5 : !llvm.i16
  }
  llvm.func @lua_set_type(%arg0: !llvm<"{ i16, i16, { i64 } }*">, %arg1: !llvm.i16) {
    %0 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %2 = llvm.alloca %1 x !llvm<"{ i16, i16, { i64 } }*"> : (!llvm.i32) -> !llvm<"{ i16, i16, { i64 } }**">
    %3 = llvm.alloca %1 x !llvm.i16 : (!llvm.i32) -> !llvm<"i16*">
    llvm.store %arg0, %2 : !llvm<"{ i16, i16, { i64 } }**">
    llvm.store %arg1, %3 : !llvm<"i16*">
    %4 = llvm.load %3 : !llvm<"i16*">
    %5 = llvm.load %2 : !llvm<"{ i16, i16, { i64 } }**">
    %6 = llvm.getelementptr %5[%0, %0] : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i32, !llvm.i32) -> !llvm<"i16*">
    llvm.store %4, %6 : !llvm<"i16*">
    llvm.return
  }
  llvm.func @lua_get_int64_val(%arg0: !llvm<"{ i16, i16, { i64 } }*">) -> !llvm.i64 {
    %0 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %1 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %2 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %3 = llvm.alloca %2 x !llvm<"{ i16, i16, { i64 } }*"> : (!llvm.i32) -> !llvm<"{ i16, i16, { i64 } }**">
    llvm.store %arg0, %3 : !llvm<"{ i16, i16, { i64 } }**">
    %4 = llvm.load %3 : !llvm<"{ i16, i16, { i64 } }**">
    %5 = llvm.getelementptr %4[%1, %0] : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i64 }*">
    %6 = llvm.bitcast %5 : !llvm<"{ i64 }*"> to !llvm<"i64*">
    %7 = llvm.load %6 : !llvm<"i64*">
    llvm.return %7 : !llvm.i64
  }
  llvm.func @lua_set_int64_val(%arg0: !llvm<"{ i16, i16, { i64 } }*">, %arg1: !llvm.i64) {
    %0 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %1 = llvm.mlir.constant(0 : i16) : !llvm.i16
    %2 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %3 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %4 = llvm.alloca %3 x !llvm<"{ i16, i16, { i64 } }*"> : (!llvm.i32) -> !llvm<"{ i16, i16, { i64 } }**">
    %5 = llvm.alloca %3 x !llvm.i64 : (!llvm.i32) -> !llvm<"i64*">
    llvm.store %arg0, %4 : !llvm<"{ i16, i16, { i64 } }**">
    llvm.store %arg1, %5 : !llvm<"i64*">
    %6 = llvm.load %4 : !llvm<"{ i16, i16, { i64 } }**">
    %7 = llvm.getelementptr %6[%2, %3] : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i32, !llvm.i32) -> !llvm<"i16*">
    llvm.store %1, %7 : !llvm<"i16*">
    %8 = llvm.load %5 : !llvm<"i64*">
    %9 = llvm.load %4 : !llvm<"{ i16, i16, { i64 } }**">
    %10 = llvm.getelementptr %9[%2, %0] : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i64 }*">
    %11 = llvm.bitcast %10 : !llvm<"{ i64 }*"> to !llvm<"i64*">
    llvm.store %8, %11 : !llvm<"i64*">
    llvm.return
  }
  llvm.func @lua_get_double_val(%arg0: !llvm<"{ i16, i16, { i64 } }*">) -> !llvm.double {
    %0 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %1 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %2 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %3 = llvm.alloca %2 x !llvm<"{ i16, i16, { i64 } }*"> : (!llvm.i32) -> !llvm<"{ i16, i16, { i64 } }**">
    llvm.store %arg0, %3 : !llvm<"{ i16, i16, { i64 } }**">
    %4 = llvm.load %3 : !llvm<"{ i16, i16, { i64 } }**">
    %5 = llvm.getelementptr %4[%1, %0] : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i64 }*">
    %6 = llvm.bitcast %5 : !llvm<"{ i64 }*"> to !llvm<"double*">
    %7 = llvm.load %6 : !llvm<"double*">
    llvm.return %7 : !llvm.double
  }
  llvm.func @lua_set_double_val(%arg0: !llvm<"{ i16, i16, { i64 } }*">, %arg1: !llvm.double) {
    %0 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %1 = llvm.mlir.constant(1 : i16) : !llvm.i16
    %2 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %3 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %4 = llvm.alloca %3 x !llvm<"{ i16, i16, { i64 } }*"> : (!llvm.i32) -> !llvm<"{ i16, i16, { i64 } }**">
    %5 = llvm.alloca %3 x !llvm.double : (!llvm.i32) -> !llvm<"double*">
    llvm.store %arg0, %4 : !llvm<"{ i16, i16, { i64 } }**">
    llvm.store %arg1, %5 : !llvm<"double*">
    %6 = llvm.load %4 : !llvm<"{ i16, i16, { i64 } }**">
    %7 = llvm.getelementptr %6[%2, %3] : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i32, !llvm.i32) -> !llvm<"i16*">
    llvm.store %1, %7 : !llvm<"i16*">
    %8 = llvm.load %5 : !llvm<"double*">
    %9 = llvm.load %4 : !llvm<"{ i16, i16, { i64 } }**">
    %10 = llvm.getelementptr %9[%2, %0] : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i64 }*">
    %11 = llvm.bitcast %10 : !llvm<"{ i64 }*"> to !llvm<"double*">
    llvm.store %8, %11 : !llvm<"double*">
    llvm.return
  }
  llvm.func @lua_get_fcn_addr(%arg0: !llvm<"{ i16, i16, { i64 } }*">) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }* ({ i64, i64, { i16, i16, { i64 } }** }*)*"> {
    %0 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %1 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %2 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %3 = llvm.alloca %2 x !llvm<"{ i16, i16, { i64 } }*"> : (!llvm.i32) -> !llvm<"{ i16, i16, { i64 } }**">
    llvm.store %arg0, %3 : !llvm<"{ i16, i16, { i64 } }**">
    %4 = llvm.load %3 : !llvm<"{ i16, i16, { i64 } }**">
    %5 = llvm.getelementptr %4[%1, %0] : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i64 }*">
    %6 = llvm.bitcast %5 : !llvm<"{ i64 }*"> to !llvm<"{ { i64, i64, { i16, i16, { i64 } }** }* ({ i64, i64, { i16, i16, { i64 } }** }*)* }**">
    %7 = llvm.load %6 : !llvm<"{ { i64, i64, { i16, i16, { i64 } }** }* ({ i64, i64, { i16, i16, { i64 } }** }*)* }**">
    %8 = llvm.bitcast %7 : !llvm<"{ { i64, i64, { i16, i16, { i64 } }** }* ({ i64, i64, { i16, i16, { i64 } }** }*)* }*"> to !llvm<"{ i64, i64, { i16, i16, { i64 } }** }* ({ i64, i64, { i16, i16, { i64 } }** }*)**">
    %9 = llvm.load %8 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }* ({ i64, i64, { i16, i16, { i64 } }** }*)**">
    llvm.return %9 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }* ({ i64, i64, { i16, i16, { i64 } }** }*)*">
  }
  llvm.func @lua_get_value_union(%arg0: !llvm<"{ i16, i16, { i64 } }*">) -> !llvm.i64 {
    %0 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %1 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %2 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %3 = llvm.alloca %2 x !llvm<"{ i16, i16, { i64 } }*"> : (!llvm.i32) -> !llvm<"{ i16, i16, { i64 } }**">
    llvm.store %arg0, %3 : !llvm<"{ i16, i16, { i64 } }**">
    %4 = llvm.load %3 : !llvm<"{ i16, i16, { i64 } }**">
    %5 = llvm.getelementptr %4[%1, %0] : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i64 }*">
    %6 = llvm.bitcast %5 : !llvm<"{ i64 }*"> to !llvm<"i64*">
    %7 = llvm.load %6 : !llvm<"i64*">
    llvm.return %7 : !llvm.i64
  }
  llvm.func @lua_set_value_union(%arg0: !llvm<"{ i16, i16, { i64 } }*">, %arg1: !llvm.i64) {
    %0 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %1 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %2 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %3 = llvm.alloca %2 x !llvm<"{ i16, i16, { i64 } }*"> : (!llvm.i32) -> !llvm<"{ i16, i16, { i64 } }**">
    %4 = llvm.alloca %2 x !llvm.i64 : (!llvm.i32) -> !llvm<"i64*">
    llvm.store %arg0, %3 : !llvm<"{ i16, i16, { i64 } }**">
    llvm.store %arg1, %4 : !llvm<"i64*">
    %5 = llvm.load %4 : !llvm<"i64*">
    %6 = llvm.load %3 : !llvm<"{ i16, i16, { i64 } }**">
    %7 = llvm.getelementptr %6[%1, %0] : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i64 }*">
    %8 = llvm.bitcast %7 : !llvm<"{ i64 }*"> to !llvm<"i64*">
    llvm.store %5, %8 : !llvm<"i64*">
    llvm.return
  }
  llvm.func @lua_is_int(%arg0: !llvm<"{ i16, i16, { i64 } }*">) -> !llvm.i1 {
    %0 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %2 = llvm.alloca %1 x !llvm<"{ i16, i16, { i64 } }*"> : (!llvm.i32) -> !llvm<"{ i16, i16, { i64 } }**">
    llvm.store %arg0, %2 : !llvm<"{ i16, i16, { i64 } }**">
    %3 = llvm.load %2 : !llvm<"{ i16, i16, { i64 } }**">
    %4 = llvm.getelementptr %3[%0, %1] : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i32, !llvm.i32) -> !llvm<"i16*">
    %5 = llvm.load %4 : !llvm<"i16*">
    %6 = llvm.sext %5 : !llvm.i16 to !llvm.i32
    %7 = llvm.icmp "eq" %6, %0 : !llvm.i32
    llvm.return %7 : !llvm.i1
  }
  llvm.func @lua_new_pack(%arg0: !llvm.i64) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*"> {
    %0 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %1 = llvm.mlir.constant(8 : i64) : !llvm.i64
    %2 = llvm.mlir.constant(0 : i64) : !llvm.i64
    %3 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %4 = llvm.mlir.constant(24 : i64) : !llvm.i64
    %5 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %6 = llvm.alloca %5 x !llvm.i64 : (!llvm.i32) -> !llvm<"i64*">
    %7 = llvm.alloca %5 x !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*"> : (!llvm.i32) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    llvm.store %arg0, %6 : !llvm<"i64*">
    %8 = llvm.call @malloc(%4) : (!llvm.i64) -> !llvm<"i8*">
    %9 = llvm.bitcast %8 : !llvm<"i8*"> to !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">
    llvm.store %9, %7 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    %10 = llvm.load %7 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    %11 = llvm.getelementptr %10[%3, %3] : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %2, %11 : !llvm<"i64*">
    %12 = llvm.load %7 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    %13 = llvm.getelementptr %12[%3, %5] : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    llvm.store %2, %13 : !llvm<"i64*">
    %14 = llvm.load %6 : !llvm<"i64*">
    %15 = llvm.mul %1, %14 : !llvm.i64
    %16 = llvm.call @malloc(%15) : (!llvm.i64) -> !llvm<"i8*">
    %17 = llvm.bitcast %16 : !llvm<"i8*"> to !llvm<"{ i16, i16, { i64 } }**">
    %18 = llvm.load %7 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    %19 = llvm.getelementptr %18[%3, %0] : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i16, i16, { i64 } }***">
    llvm.store %17, %19 : !llvm<"{ i16, i16, { i64 } }***">
    %20 = llvm.load %7 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    llvm.return %20 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">
  }
  llvm.func @lua_delete_pack(%arg0: !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) {
    %0 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %1 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %2 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %3 = llvm.alloca %2 x !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*"> : (!llvm.i32) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    llvm.store %arg0, %3 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    %4 = llvm.load %3 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    %5 = llvm.getelementptr %4[%1, %0] : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i16, i16, { i64 } }***">
    %6 = llvm.load %5 : !llvm<"{ i16, i16, { i64 } }***">
    %7 = llvm.bitcast %6 : !llvm<"{ i16, i16, { i64 } }**"> to !llvm<"i8*">
    llvm.call @free(%7) : (!llvm<"i8*">) -> ()
    %8 = llvm.load %3 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    %9 = llvm.bitcast %8 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*"> to !llvm<"i8*">
    llvm.call @free(%9) : (!llvm<"i8*">) -> ()
    llvm.return
  }
  llvm.func @free(!llvm<"i8*">)
  llvm.func @lua_pack_push(%arg0: !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, %arg1: !llvm<"{ i16, i16, { i64 } }*">) {
    %0 = llvm.mlir.constant(1 : i64) : !llvm.i64
    %1 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %2 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %3 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %4 = llvm.alloca %3 x !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*"> : (!llvm.i32) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    %5 = llvm.alloca %3 x !llvm<"{ i16, i16, { i64 } }*"> : (!llvm.i32) -> !llvm<"{ i16, i16, { i64 } }**">
    llvm.store %arg0, %4 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    llvm.store %arg1, %5 : !llvm<"{ i16, i16, { i64 } }**">
    %6 = llvm.load %5 : !llvm<"{ i16, i16, { i64 } }**">
    %7 = llvm.load %4 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    %8 = llvm.getelementptr %7[%2, %1] : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i16, i16, { i64 } }***">
    %9 = llvm.load %8 : !llvm<"{ i16, i16, { i64 } }***">
    %10 = llvm.load %4 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    %11 = llvm.getelementptr %10[%2, %2] : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %12 = llvm.load %11 : !llvm<"i64*">
    %13 = llvm.add %12, %0 : !llvm.i64
    llvm.store %13, %11 : !llvm<"i64*">
    %14 = llvm.getelementptr %9[%12] : (!llvm<"{ i16, i16, { i64 } }**">, !llvm.i64) -> !llvm<"{ i16, i16, { i64 } }**">
    llvm.store %6, %14 : !llvm<"{ i16, i16, { i64 } }**">
    llvm.return
  }
  llvm.func @lua_pack_pull_one(%arg0: !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) -> !llvm<"{ i16, i16, { i64 } }*"> {
    %0 = llvm.mlir.constant(1 : i64) : !llvm.i64
    %1 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %2 = llvm.mlir.constant(0 : i16) : !llvm.i16
    %3 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %4 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %5 = llvm.alloca %4 x !llvm<"{ i16, i16, { i64 } }*"> : (!llvm.i32) -> !llvm<"{ i16, i16, { i64 } }**">
    %6 = llvm.alloca %4 x !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*"> : (!llvm.i32) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    %7 = llvm.alloca %4 x !llvm<"{ i16, i16, { i64 } }*"> : (!llvm.i32) -> !llvm<"{ i16, i16, { i64 } }**">
    llvm.store %arg0, %6 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    %8 = llvm.load %6 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    %9 = llvm.getelementptr %8[%3, %4] : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %10 = llvm.load %9 : !llvm<"i64*">
    %11 = llvm.load %6 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    %12 = llvm.getelementptr %11[%3, %3] : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %13 = llvm.load %12 : !llvm<"i64*">
    %14 = llvm.icmp "eq" %10, %13 : !llvm.i64
    llvm.cond_br %14, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %15 = llvm.call @lua_alloc() : () -> !llvm<"{ i16, i16, { i64 } }*">
    llvm.store %15, %7 : !llvm<"{ i16, i16, { i64 } }**">
    %16 = llvm.load %7 : !llvm<"{ i16, i16, { i64 } }**">
    llvm.call @lua_set_type(%16, %2) : (!llvm<"{ i16, i16, { i64 } }*">, !llvm.i16) -> ()
    %17 = llvm.load %7 : !llvm<"{ i16, i16, { i64 } }**">
    llvm.store %17, %5 : !llvm<"{ i16, i16, { i64 } }**">
    llvm.br ^bb3
  ^bb2:  // pred: ^bb0
    %18 = llvm.load %6 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    %19 = llvm.getelementptr %18[%3, %1] : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i16, i16, { i64 } }***">
    %20 = llvm.load %19 : !llvm<"{ i16, i16, { i64 } }***">
    %21 = llvm.load %6 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    %22 = llvm.getelementptr %21[%3, %4] : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %23 = llvm.load %22 : !llvm<"i64*">
    %24 = llvm.add %23, %0 : !llvm.i64
    llvm.store %24, %22 : !llvm<"i64*">
    %25 = llvm.getelementptr %20[%23] : (!llvm<"{ i16, i16, { i64 } }**">, !llvm.i64) -> !llvm<"{ i16, i16, { i64 } }**">
    %26 = llvm.load %25 : !llvm<"{ i16, i16, { i64 } }**">
    llvm.store %26, %5 : !llvm<"{ i16, i16, { i64 } }**">
    llvm.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    %27 = llvm.load %5 : !llvm<"{ i16, i16, { i64 } }**">
    llvm.return %27 : !llvm<"{ i16, i16, { i64 } }*">
  }
  llvm.func @lua_pack_push_all(%arg0: !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, %arg1: !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) {
    %0 = llvm.mlir.constant(1 : i64) : !llvm.i64
    %1 = llvm.mlir.constant(2 : i32) : !llvm.i32
    %2 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %3 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %4 = llvm.alloca %3 x !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*"> : (!llvm.i32) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    %5 = llvm.alloca %3 x !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*"> : (!llvm.i32) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    llvm.store %arg0, %4 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    llvm.store %arg1, %5 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    llvm.br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb2
    %6 = llvm.load %5 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    %7 = llvm.getelementptr %6[%2, %3] : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %8 = llvm.load %7 : !llvm<"i64*">
    %9 = llvm.load %5 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    %10 = llvm.getelementptr %9[%2, %2] : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %11 = llvm.load %10 : !llvm<"i64*">
    %12 = llvm.icmp "ne" %8, %11 : !llvm.i64
    llvm.cond_br %12, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %13 = llvm.load %4 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    %14 = llvm.load %5 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    %15 = llvm.getelementptr %14[%2, %1] : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"{ i16, i16, { i64 } }***">
    %16 = llvm.load %15 : !llvm<"{ i16, i16, { i64 } }***">
    %17 = llvm.load %5 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    %18 = llvm.getelementptr %17[%2, %3] : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %19 = llvm.load %18 : !llvm<"i64*">
    %20 = llvm.add %19, %0 : !llvm.i64
    llvm.store %20, %18 : !llvm<"i64*">
    %21 = llvm.getelementptr %16[%19] : (!llvm<"{ i16, i16, { i64 } }**">, !llvm.i64) -> !llvm<"{ i16, i16, { i64 } }**">
    %22 = llvm.load %21 : !llvm<"{ i16, i16, { i64 } }**">
    llvm.call @lua_pack_push(%13, %22) : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm<"{ i16, i16, { i64 } }*">) -> ()
    llvm.br ^bb1
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func @lua_pack_get_size(%arg0: !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">) -> !llvm.i64 {
    %0 = llvm.mlir.constant(0 : i32) : !llvm.i32
    %1 = llvm.mlir.constant(1 : i32) : !llvm.i32
    %2 = llvm.alloca %1 x !llvm<"{ i64, i64, { i16, i16, { i64 } }** }*"> : (!llvm.i32) -> !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    llvm.store %arg0, %2 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    %3 = llvm.load %2 : !llvm<"{ i64, i64, { i16, i16, { i64 } }** }**">
    %4 = llvm.getelementptr %3[%0, %0] : (!llvm<"{ i64, i64, { i16, i16, { i64 } }** }*">, !llvm.i32, !llvm.i32) -> !llvm<"i64*">
    %5 = llvm.load %4 : !llvm<"i64*">
    llvm.return %5 : !llvm.i64
  }
  llvm.func @lua_builtin_print() -> !llvm<"{ i16, i16, { i64 } }*"> {
    %0 = llvm.mlir.addressof @builtin_print : !llvm<"{ i16, i16, { i64 } }**">
    %1 = llvm.load %0 : !llvm<"{ i16, i16, { i64 } }**">
    llvm.return %1 : !llvm<"{ i16, i16, { i64 } }*">
  }
}
