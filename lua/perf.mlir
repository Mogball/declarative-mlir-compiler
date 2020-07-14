

module {
  func @lua_main() -> !lua.pack {
    %f = constant @lua_anon_fcn_1 : (!lua.capture, !lua.pack) -> !lua.pack
    %f_0 = constant @lua_anon_fcn_0 : (!lua.capture, !lua.pack) -> !lua.pack
    %cst = constant 4.000000e+00 : f64
    %cst_1 = constant 1.800000e+01 : f64
    %cst_2 = constant 1.000000e+00 : f64
    %cst_3 = constant 2.000000e+00 : f64
    %cst_4 = constant 0.000000e+00 : f64
    %c5_i32 = constant 5 : i32
    %c4_i32 = constant 4 : i32
    %c1_i32 = constant 1 : i32
    %c2_i32 = constant 2 : i32
    %c3_i32 = constant 3 : i32
    %c0_i32 = constant 0 : i32
    %0 = lua.builtin "print"
    %1 = lua.nil
    %2 = "luac.new_capture"(%c1_i32) : (i32) -> !lua.capture
    %3 = luac.get_ref %1
    "luac.add_capture"(%2, %3, %c0_i32) : (!lua.capture, !luac.ref, i32) -> ()
    %4 = "luac.make_fcn"(%f, %2) : ((!lua.capture, !lua.pack) -> !lua.pack, !lua.capture) -> !lua.val
    "luac.copy"(%3, %4) : (!luac.ref, !lua.val) -> ()
    "luac.add_capture"(%2, %3, %c0_i32) : (!lua.capture, !luac.ref, i32) -> ()
    %5 = "luac.make_fcn"(%f_0, %2) : ((!lua.capture, !lua.pack) -> !lua.pack, !lua.capture) -> !lua.val
    "luac.copy"(%3, %5) : (!luac.ref, !lua.val) -> ()
    %6 = luac.wrap_real %cst
    %7 = luac.wrap_real %cst_1
    %8 = luac.wrap_real %cst_2
    %9 = luac.add(%7, %8)
    %10 = luac.wrap_real %cst_4
    %11 = "luac.get_arg_pack"(%c2_i32) : (i32) -> !lua.pack
    "luac.pack_insert"(%11, %10, %c0_i32) : (!lua.pack, !lua.val, i32) -> ()
    "luac.pack_insert"(%11, %9, %c1_i32) : (!lua.pack, !lua.val, i32) -> ()
    %12 = luac.get_fcn_addr %1
    %13 = luac.get_capture_pack %1[]
    %14 = call_indirect %12(%13, %11) : (!lua.capture, !lua.pack) -> !lua.pack
    %15 = "luaopt.unpack_unsafe"(%14) : (!lua.pack) -> !lua.val
    %16 = luac.load_string @lua_anon_string_0
    %17 = luac.load_string @lua_anon_string_1
    %18 = "luac.get_arg_pack"(%c1_i32) : (i32) -> !lua.pack
    "luac.pack_insert"(%18, %15, %c0_i32) : (!lua.pack, !lua.val, i32) -> ()
    %19 = call_indirect %12(%13, %18) : (!lua.capture, !lua.pack) -> !lua.pack
    %20 = "luac.pack_get_size"(%19) : (!lua.pack) -> i32
    %21 = addi %20, %c3_i32 : i32
    %22 = "luac.get_arg_pack"(%21) : (i32) -> !lua.pack
    "luac.pack_insert"(%22, %16, %c0_i32) : (!lua.pack, !lua.val, i32) -> ()
    "luac.pack_insert"(%22, %9, %c1_i32) : (!lua.pack, !lua.val, i32) -> ()
    "luac.pack_insert"(%22, %17, %c2_i32) : (!lua.pack, !lua.val, i32) -> ()
    "luac.pack_insert_all"(%22, %19, %c3_i32) : (!lua.pack, !lua.pack, i32) -> ()
    %23 = luac.get_fcn_addr %0
    %24 = luac.get_capture_pack %0[]
    %25 = call_indirect %23(%24, %22) : (!lua.capture, !lua.pack) -> !lua.pack
    "luac.pack_insert"(%11, %10, %c0_i32) : (!lua.pack, !lua.val, i32) -> ()
    "luac.pack_insert"(%11, %7, %c1_i32) : (!lua.pack, !lua.val, i32) -> ()
    %26 = call_indirect %12(%13, %11) : (!lua.capture, !lua.pack) -> !lua.pack
    %27 = "luaopt.unpack_unsafe"(%26) : (!lua.pack) -> !lua.val
    %28 = luac.wrap_real %cst_3
    %29 = luac.neg %8
    %30 = luac.load_string @lua_anon_string_2
    %31 = luac.get_double_val %28
    %32 = luac.get_double_val %6
    %33 = luac.wrap_real %32
    br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb5
    %34 = luac.le(%33, %7)
    %35 = luac.get_bool_val %34
    cond_br %35, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    %36 = luac.sub(%7, %33)
    %37 = luac.add(%36, %6)
    %38 = luac.pow(%28, %37)
    %39 = luac.wrap_real %cst_4
    %40 = luac.get_double_val %8
    %41 = luac.wrap_real %40
    br ^bb3
  ^bb3:  // 2 preds: ^bb2, ^bb4
    %42 = luac.le(%41, %38)
    %43 = luac.get_bool_val %42
    cond_br %43, ^bb4, ^bb5
  ^bb4:  // pred: ^bb3
    "luac.pack_insert"(%11, %8, %c0_i32) : (!lua.pack, !lua.val, i32) -> ()
    "luac.pack_insert"(%11, %33, %c1_i32) : (!lua.pack, !lua.val, i32) -> ()
    %44 = call_indirect %12(%13, %11) : (!lua.capture, !lua.pack) -> !lua.pack
    %45 = call_indirect %12(%13, %44) : (!lua.capture, !lua.pack) -> !lua.pack
    %46 = "luaopt.unpack_unsafe"(%45) : (!lua.pack) -> !lua.val
    %47 = luac.add(%39, %46)
    "luac.pack_insert"(%11, %29, %c0_i32) : (!lua.pack, !lua.val, i32) -> ()
    "luac.pack_insert"(%11, %33, %c1_i32) : (!lua.pack, !lua.val, i32) -> ()
    %48 = call_indirect %12(%13, %11) : (!lua.capture, !lua.pack) -> !lua.pack
    %49 = call_indirect %12(%13, %48) : (!lua.capture, !lua.pack) -> !lua.pack
    %50 = "luaopt.unpack_unsafe"(%49) : (!lua.pack) -> !lua.val
    %51 = luac.add(%47, %50)
    %52 = luac.get_ref %39
    "luac.copy"(%52, %51) : (!luac.ref, !lua.val) -> ()
    %53 = luac.get_double_val %41
    %54 = addf %53, %40 : f64
    %55 = luac.get_ref %41
    luac.set_double_val %55 = %54
    br ^bb3
  ^bb5:  // pred: ^bb3
    %56 = luac.mul(%38, %28)
    %57 = "luac.get_arg_pack"(%c5_i32) : (i32) -> !lua.pack
    "luac.pack_insert"(%57, %56, %c0_i32) : (!lua.pack, !lua.val, i32) -> ()
    "luac.pack_insert"(%57, %30, %c1_i32) : (!lua.pack, !lua.val, i32) -> ()
    "luac.pack_insert"(%57, %33, %c2_i32) : (!lua.pack, !lua.val, i32) -> ()
    "luac.pack_insert"(%57, %17, %c3_i32) : (!lua.pack, !lua.val, i32) -> ()
    "luac.pack_insert"(%57, %39, %c4_i32) : (!lua.pack, !lua.val, i32) -> ()
    %58 = call_indirect %23(%24, %57) : (!lua.capture, !lua.pack) -> !lua.pack
    %59 = luac.get_double_val %33
    %60 = addf %59, %31 : f64
    %61 = luac.get_ref %33
    luac.set_double_val %61 = %60
    br ^bb1
  ^bb6:  // pred: ^bb1
    %62 = luac.load_string @lua_anon_string_3
    "luac.pack_insert"(%18, %27, %c0_i32) : (!lua.pack, !lua.val, i32) -> ()
    %63 = call_indirect %12(%13, %18) : (!lua.capture, !lua.pack) -> !lua.pack
    %64 = "luac.pack_get_size"(%63) : (!lua.pack) -> i32
    %65 = addi %64, %c3_i32 : i32
    %66 = "luac.get_arg_pack"(%65) : (i32) -> !lua.pack
    "luac.pack_insert"(%66, %62, %c0_i32) : (!lua.pack, !lua.val, i32) -> ()
    "luac.pack_insert"(%66, %7, %c1_i32) : (!lua.pack, !lua.val, i32) -> ()
    "luac.pack_insert"(%66, %17, %c2_i32) : (!lua.pack, !lua.val, i32) -> ()
    "luac.pack_insert_all"(%66, %63, %c3_i32) : (!lua.pack, !lua.pack, i32) -> ()
    %67 = call_indirect %23(%24, %66) : (!lua.capture, !lua.pack) -> !lua.pack
    %68 = "luac.get_ret_pack"(%c0_i32) : (i32) -> !lua.pack
    return %68 : !lua.pack
  }
  func @lua_anon_fcn_0(%arg0: !lua.capture, %arg1: !lua.pack) -> !lua.pack {
    %c1_i64 = constant 1 : i64
    %c2_i64 = constant 2 : i64
    %c0_i64 = constant 0 : i64
    %c1_i32 = constant 1 : i32
    %c0_i32 = constant 0 : i32
    %0 = "luac.get_capture"(%arg0, %c0_i32) : (!lua.capture, i32) -> !luac.ref
    %1 = "luac.dec_ref"(%0) : (!luac.ref) -> !lua.val
    %2 = "luac.pack_get"(%arg1, %c0_i32) : (!lua.pack, i32) -> !lua.val
    %3 = "luaopt.table_get_prealloc"(%2, %c1_i64) : (!lua.val, i64) -> !lua.val
    %4 = luac.convert_bool_like %3
    cond_br %4, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %5 = "luaopt.table_get_prealloc"(%2, %c0_i64) : (!lua.val, i64) -> !lua.val
    %6 = "luac.get_arg_pack"(%c1_i32) : (i32) -> !lua.pack
    "luac.pack_insert"(%6, %3, %c0_i32) : (!lua.pack, !lua.val, i32) -> ()
    %7 = luac.get_fcn_addr %1
    %8 = luac.get_capture_pack %1[]
    %9 = call_indirect %7(%8, %6) : (!lua.capture, !lua.pack) -> !lua.pack
    %10 = "luaopt.unpack_unsafe"(%9) : (!lua.pack) -> !lua.val
    %11 = luac.add(%5, %10)
    %12 = "luaopt.table_get_prealloc"(%2, %c2_i64) : (!lua.val, i64) -> !lua.val
    "luac.pack_insert"(%6, %12, %c0_i32) : (!lua.pack, !lua.val, i32) -> ()
    %13 = call_indirect %7(%8, %6) : (!lua.capture, !lua.pack) -> !lua.pack
    %14 = "luaopt.unpack_unsafe"(%13) : (!lua.pack) -> !lua.val
    %15 = luac.sub(%11, %14)
    %16 = "luac.get_ret_pack"(%c1_i32) : (i32) -> !lua.pack
    "luac.pack_insert"(%16, %15, %c0_i32) : (!lua.pack, !lua.val, i32) -> ()
    return %16 : !lua.pack
  ^bb2:  // pred: ^bb0
    %17 = "luaopt.table_get_prealloc"(%2, %c0_i64) : (!lua.val, i64) -> !lua.val
    %18 = "luac.get_ret_pack"(%c1_i32) : (i32) -> !lua.pack
    "luac.pack_insert"(%18, %17, %c0_i32) : (!lua.pack, !lua.val, i32) -> ()
    return %18 : !lua.pack
  }
  func @lua_anon_fcn_1(%arg0: !lua.capture, %arg1: !lua.pack) -> !lua.pack {
    %c1_i64 = constant 1 : i64
    %c2_i64 = constant 2 : i64
    %c0_i64 = constant 0 : i64
    %cst = constant 0.000000e+00 : f64
    %cst_0 = constant 1.000000e+00 : f64
    %c2_i32 = constant 2 : i32
    %c1_i32 = constant 1 : i32
    %c0_i32 = constant 0 : i32
    %0 = "luac.get_capture"(%arg0, %c0_i32) : (!lua.capture, i32) -> !luac.ref
    %1 = "luac.dec_ref"(%0) : (!luac.ref) -> !lua.val
    %2 = "luac.pack_get"(%arg1, %c0_i32) : (!lua.pack, i32) -> !lua.val
    %3 = "luac.pack_get"(%arg1, %c1_i32) : (!lua.pack, i32) -> !lua.val
    %4 = luac.wrap_real %cst
    %5 = luac.gt(%3, %4)
    %6 = luac.get_bool_val %5
    cond_br %6, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %7 = luac.add(%2, %2)
    %8 = luac.wrap_real %cst_0
    %9 = luac.sub(%3, %8)
    %10 = luac.get_ref %3
    "luac.copy"(%10, %9) : (!luac.ref, !lua.val) -> ()
    %11 = luac.sub(%7, %8)
    %12 = "luac.get_arg_pack"(%c2_i32) : (i32) -> !lua.pack
    "luac.pack_insert"(%12, %11, %c0_i32) : (!lua.pack, !lua.val, i32) -> ()
    "luac.pack_insert"(%12, %3, %c1_i32) : (!lua.pack, !lua.val, i32) -> ()
    %13 = luac.get_fcn_addr %1
    %14 = luac.get_capture_pack %1[]
    %15 = call_indirect %13(%14, %12) : (!lua.capture, !lua.pack) -> !lua.pack
    %16 = "luaopt.unpack_unsafe"(%15) : (!lua.pack) -> !lua.val
    "luac.pack_insert"(%12, %7, %c0_i32) : (!lua.pack, !lua.val, i32) -> ()
    "luac.pack_insert"(%12, %3, %c1_i32) : (!lua.pack, !lua.val, i32) -> ()
    %17 = call_indirect %13(%14, %12) : (!lua.capture, !lua.pack) -> !lua.pack
    %18 = "luaopt.unpack_unsafe"(%17) : (!lua.pack) -> !lua.val
    %19 = lua.table
    "luaopt.table_set_prealloc"(%19, %c0_i64, %2) : (!lua.val, i64, !lua.val) -> ()
    "luaopt.table_set_prealloc"(%19, %c1_i64, %16) : (!lua.val, i64, !lua.val) -> ()
    "luaopt.table_set_prealloc"(%19, %c2_i64, %18) : (!lua.val, i64, !lua.val) -> ()
    %20 = "luac.get_ret_pack"(%c1_i32) : (i32) -> !lua.pack
    "luac.pack_insert"(%20, %19, %c0_i32) : (!lua.pack, !lua.val, i32) -> ()
    return %20 : !lua.pack
  ^bb2:  // pred: ^bb0
    %21 = lua.table
    "luaopt.table_set_prealloc"(%21, %c0_i64, %2) : (!lua.val, i64, !lua.val) -> ()
    %22 = "luac.get_ret_pack"(%c1_i32) : (i32) -> !lua.pack
    "luac.pack_insert"(%22, %21, %c0_i32) : (!lua.pack, !lua.val, i32) -> ()
    return %22 : !lua.pack
  }
  luac.global_string @lua_anon_string_0 = "stretch tree of depth"
  luac.global_string @lua_anon_string_1 = "check:"
  luac.global_string @lua_anon_string_2 = "trees of depth"
  luac.global_string @lua_anon_string_3 = "long lived tree of depth"
}
