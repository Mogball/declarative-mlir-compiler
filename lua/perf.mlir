

module {
  func @lua_main() -> !lua.pack {
    %f = constant @lua_anon_fcn_1 : (!lua.capture, !lua.pack) -> !lua.pack
    %f_0 = constant @lua_anon_fcn_0 : (!lua.capture, !lua.pack) -> !lua.pack
    %0 = lua.builtin "print"
    %1 = lua.alloc "ItemCheck"
    %2 = lua.alloc "BottomUpTree"
    %3 = "lua.make_capture"(%2) : (!lua.val) -> !lua.capture
    %4 = "luac.make_fcn"(%f, %3) : ((!lua.capture, !lua.pack) -> !lua.pack, !lua.capture) -> !lua.val
    lua.copy %2 = %4
    %5 = "lua.make_capture"(%1) : (!lua.val) -> !lua.capture
    %6 = "luac.make_fcn"(%f_0, %5) : ((!lua.capture, !lua.pack) -> !lua.pack, !lua.capture) -> !lua.val
    lua.copy %1 = %6
    %7 = "luaopt.const_number"() {value = 4.000000e+00 : f64} : () -> !lua.val
    %8 = "luaopt.const_number"() {value = 1.800000e+01 : f64} : () -> !lua.val
    %9 = "luaopt.const_number"() {value = 1.000000e+00 : f64} : () -> !lua.val
    %10 = lua.binary %8 "+" %9
    %11 = "luaopt.const_number"() {value = 0.000000e+00 : f64} : () -> !lua.val
    %12 = lua.concat(%11, %10) : (!lua.val, !lua.val) -> !lua.pack {operand_segment_sizes = dense<[2, 0]> : vector<2xi64>}
    %13 = lua.call %2(%12)
    %14 = lua.unpack %13 : (!lua.pack) -> !lua.val
    %15 = lua.get_string "stretch tree of depth"
    %16 = lua.get_string "check:"
    %17 = lua.concat(%14) : (!lua.val) -> !lua.pack {operand_segment_sizes = dense<[1, 0]> : vector<2xi64>}
    %18 = lua.call %1(%17)
    %19 = lua.concat(%15, %10, %16, %18) : (!lua.val, !lua.val, !lua.val, !lua.pack) -> !lua.pack {operand_segment_sizes = dense<[3, 1]> : vector<2xi64>}
    %20 = lua.call %0(%19)
    %21 = lua.concat(%11, %8) : (!lua.val, !lua.val) -> !lua.pack {operand_segment_sizes = dense<[2, 0]> : vector<2xi64>}
    %22 = lua.call %2(%21)
    %23 = lua.unpack %22 : (!lua.pack) -> !lua.val
    %24 = "luaopt.const_number"() {value = 2.000000e+00 : f64} : () -> !lua.val
    %25 = lua.unary "-" %9
    %26 = lua.get_string "trees of depth"
    %27 = luac.get_double_val %24
    %28 = luac.get_double_val %7
    %29 = luac.wrap_real %28
    br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb5
    %30 = luac.le(%29, %8)
    %31 = luac.convert_bool_like %30
    cond_br %31, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %32 = lua.binary %8 "-" %29
    %33 = lua.binary %32 "+" %7
    %34 = lua.binary %24 "^" %33
    %35 = lua.number 0.000000e+00 : f64
    %36 = luac.get_double_val %9
    %37 = luac.get_double_val %9
    %38 = luac.wrap_real %37
    br ^bb3
  ^bb3:  // 2 preds: ^bb2, ^bb4
    %39 = luac.le(%38, %34)
    %40 = luac.convert_bool_like %39
    cond_br %40, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    %41 = lua.concat(%9, %29) : (!lua.val, !lua.val) -> !lua.pack {operand_segment_sizes = dense<[2, 0]> : vector<2xi64>}
    %42 = lua.call %2(%41)
    %43 = lua.call %1(%42)
    %44 = lua.unpack %43 : (!lua.pack) -> !lua.val
    %45 = lua.binary %35 "+" %44
    %46 = lua.concat(%25, %29) : (!lua.val, !lua.val) -> !lua.pack {operand_segment_sizes = dense<[2, 0]> : vector<2xi64>}
    %47 = lua.call %2(%46)
    %48 = lua.call %1(%47)
    %49 = lua.unpack %48 : (!lua.pack) -> !lua.val
    %50 = lua.binary %45 "+" %49
    lua.copy %35 = %50
    %51 = luac.get_double_val %38
    %52 = addf %51, %36 : f64
    %53 = luac.get_ref %38
    luac.set_double_val %53 = %52
    br ^bb3
  ^bb5:  // pred: ^bb3
    %54 = lua.binary %34 "*" %24
    %55 = lua.concat(%54, %26, %29, %16, %35) : (!lua.val, !lua.val, !lua.val, !lua.val, !lua.val) -> !lua.pack {operand_segment_sizes = dense<[5, 0]> : vector<2xi64>}
    %56 = lua.call %0(%55)
    %57 = luac.get_double_val %29
    %58 = addf %57, %27 : f64
    %59 = luac.get_ref %29
    luac.set_double_val %59 = %58
    br ^bb1
  ^bb6:  // pred: ^bb1
    %60 = lua.get_string "long lived tree of depth"
    %61 = lua.concat(%23) : (!lua.val) -> !lua.pack {operand_segment_sizes = dense<[1, 0]> : vector<2xi64>}
    %62 = lua.call %1(%61)
    %63 = lua.concat(%60, %8, %16, %62) : (!lua.val, !lua.val, !lua.val, !lua.pack) -> !lua.pack {operand_segment_sizes = dense<[3, 1]> : vector<2xi64>}
    %64 = lua.call %0(%63)
    %65 = lua.concat() : () -> !lua.pack {operand_segment_sizes = dense<0> : vector<2xi64>}
    return %65 : !lua.pack
  }
  func @lua_anon_fcn_0(%arg0: !lua.capture, %arg1: !lua.pack) -> !lua.pack {
    %c1_i64 = constant 1 : i64
    %c2_i64 = constant 2 : i64
    %c0_i64 = constant 0 : i64
    %0 = "lua.get_captures"(%arg0) : (!lua.capture) -> !lua.val
    %1 = lua.unpack %arg1 : (!lua.pack) -> !lua.val
    %2 = "luaopt.capture_self"(%0) : (!lua.val) -> !lua.val
    %3 = "luaopt.table_get_prealloc"(%1, %c1_i64) : (!lua.val, i64) -> !lua.val
    %4 = luac.convert_bool_like %3
    cond_br %4, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %5 = "luaopt.table_get_prealloc"(%1, %c0_i64) : (!lua.val, i64) -> !lua.val
    %6 = lua.concat(%3) : (!lua.val) -> !lua.pack {operand_segment_sizes = dense<[1, 0]> : vector<2xi64>}
    %7 = lua.call %2(%6)
    %8 = lua.unpack %7 : (!lua.pack) -> !lua.val
    %9 = lua.binary %5 "+" %8
    %10 = "luaopt.table_get_prealloc"(%1, %c2_i64) : (!lua.val, i64) -> !lua.val
    %11 = lua.concat(%10) : (!lua.val) -> !lua.pack {operand_segment_sizes = dense<[1, 0]> : vector<2xi64>}
    %12 = lua.call %2(%11)
    %13 = lua.unpack %12 : (!lua.pack) -> !lua.val
    %14 = lua.binary %9 "-" %13
    %15 = lua.concat(%14) : (!lua.val) -> !lua.pack {operand_segment_sizes = dense<[1, 0]> : vector<2xi64>}
    return %15 : !lua.pack
  ^bb2:  // pred: ^bb0
    %16 = "luaopt.table_get_prealloc"(%1, %c0_i64) : (!lua.val, i64) -> !lua.val
    %17 = lua.concat(%16) : (!lua.val) -> !lua.pack {operand_segment_sizes = dense<[1, 0]> : vector<2xi64>}
    return %17 : !lua.pack
  }
  func @lua_anon_fcn_1(%arg0: !lua.capture, %arg1: !lua.pack) -> !lua.pack {
    %c1_i64 = constant 1 : i64
    %c2_i64 = constant 2 : i64
    %c0_i64 = constant 0 : i64
    %0 = "lua.get_captures"(%arg0) : (!lua.capture) -> !lua.val
    %1:2 = lua.unpack %arg1 : (!lua.pack) -> (!lua.val, !lua.val)
    %2 = "luaopt.capture_self"(%0) : (!lua.val) -> !lua.val
    %3 = "luaopt.const_number"() {value = 0.000000e+00 : f64} : () -> !lua.val
    %4 = lua.binary %1#1 ">" %3
    %5 = luac.convert_bool_like %4
    cond_br %5, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %6 = lua.binary %1#0 "+" %1#0
    %7 = "luaopt.const_number"() {value = 1.000000e+00 : f64} : () -> !lua.val
    %8 = lua.binary %1#1 "-" %7
    lua.copy %1#1 = %8
    %9 = lua.binary %6 "-" %7
    %10 = lua.concat(%9, %1#1) : (!lua.val, !lua.val) -> !lua.pack {operand_segment_sizes = dense<[2, 0]> : vector<2xi64>}
    %11 = lua.call %2(%10)
    %12 = lua.unpack %11 : (!lua.pack) -> !lua.val
    %13 = lua.concat(%6, %1#1) : (!lua.val, !lua.val) -> !lua.pack {operand_segment_sizes = dense<[2, 0]> : vector<2xi64>}
    %14 = lua.call %2(%13)
    %15 = lua.unpack %14 : (!lua.pack) -> !lua.val
    %16 = lua.table
    "luaopt.table_set_prealloc"(%16, %c0_i64, %1#0) : (!lua.val, i64, !lua.val) -> ()
    "luaopt.table_set_prealloc"(%16, %c1_i64, %12) : (!lua.val, i64, !lua.val) -> ()
    "luaopt.table_set_prealloc"(%16, %c2_i64, %15) : (!lua.val, i64, !lua.val) -> ()
    %17 = lua.concat(%16) : (!lua.val) -> !lua.pack {operand_segment_sizes = dense<[1, 0]> : vector<2xi64>}
    return %17 : !lua.pack
  ^bb2:  // pred: ^bb0
    %18 = lua.table
    "luaopt.table_set_prealloc"(%18, %c0_i64, %1#0) : (!lua.val, i64, !lua.val) -> ()
    %19 = lua.concat(%18) : (!lua.val) -> !lua.pack {operand_segment_sizes = dense<[1, 0]> : vector<2xi64>}
    return %19 : !lua.pack
  }
}
