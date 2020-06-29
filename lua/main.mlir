

module {
  func @main() {
    %c5_i64 = constant 5 : i64
    %0 = luac.alloc
    %c2_i32 = constant 2 : i32
    luac.set_int64_val %0 = %c5_i64
    luac.set_type type(%0) = %c2_i32
    %c6_i64 = constant 6 : i64
    %1 = luac.alloc
    %c2_i32_0 = constant 2 : i32
    luac.set_int64_val %1 = %c6_i64
    luac.set_type type(%1) = %c2_i32_0
    %c8_i64 = constant 8 : i64
    %2 = luac.alloc
    %c2_i32_1 = constant 2 : i32
    luac.set_int64_val %2 = %c8_i64
    luac.set_type type(%2) = %c2_i32_1
    %c9_i64 = constant 9 : i64
    %3 = luac.alloc
    %c2_i32_2 = constant 2 : i32
    luac.set_int64_val %3 = %c9_i64
    luac.set_type type(%3) = %c2_i32_2
    %4 = luac.alloc
    %c0_i32 = constant 0 : i32
    luac.set_type type(%4) = %c0_i32
    %5 = lua.builtin "print"
    %c5_i64_3 = constant 5 : i64
    %6 = luac.alloc
    %c2_i32_4 = constant 2 : i32
    luac.set_int64_val %6 = %c5_i64_3
    luac.set_type type(%6) = %c2_i32_4
    %c6_i64_5 = constant 6 : i64
    %7 = luac.alloc
    %c2_i32_6 = constant 2 : i32
    luac.set_int64_val %7 = %c6_i64_5
    luac.set_type type(%7) = %c2_i32_6
    %c2_i32_7 = constant 2 : i32
    %8 = luac.new_pack[%c2_i32_7]
    luac.pack_push(%8, [%6])
    luac.pack_push(%8, [%7])
    %9 = luac.get_fcn_addr %5
    %10 = call_indirect %9(%8) : (!lua.pack) -> !lua.pack
    luac.delete_pack %10
    %c2_i32_8 = constant 2 : i32
    %11 = luac.new_pack[%c2_i32_8]
    luac.pack_push(%11, [%0])
    luac.pack_push(%11, [%1])
    %12 = luac.get_fcn_addr %5
    %13 = call_indirect %12(%11) : (!lua.pack) -> !lua.pack
    luac.delete_pack %13
    %c3_i32 = constant 3 : i32
    %14 = luac.new_pack[%c3_i32]
    luac.pack_push(%14, [%2])
    luac.pack_push(%14, [%3])
    luac.pack_push(%14, [%4])
    %15 = luac.get_fcn_addr %5
    %16 = call_indirect %15(%14) : (!lua.pack) -> !lua.pack
    luac.delete_pack %16
    %17 = luac.add(%0, %1)
    %18 = luac.add(%17, %2)
    %19 = luac.add(%18, %3)
    %20 = luac.add(%19, %4)
    %21 = luac.mul(%1, %2)
    %22 = luac.add(%0, %21)
    %23 = luac.add(%0, %1)
    %24 = luac.mul(%23, %2)
    %c3_i32_9 = constant 3 : i32
    %25 = luac.new_pack[%c3_i32_9]
    luac.pack_push(%25, [%20])
    luac.pack_push(%25, [%22])
    luac.pack_push(%25, [%24])
    %26 = luac.get_fcn_addr %5
    %27 = call_indirect %26(%25) : (!lua.pack) -> !lua.pack
    luac.delete_pack %27
    %c1_i32 = constant 1 : i32
    %28 = luac.new_pack[%c1_i32]
    luac.pack_push(%28, [%1])
    %29 = luac.get_fcn_addr %5
    %30 = call_indirect %29(%28) : (!lua.pack) -> !lua.pack
    %c3_i32_10 = constant 3 : i32
    %31 = luac.new_pack[%c3_i32_10]
    luac.pack_push(%31, [%0])
    luac.pack_push_all(%31, %30)
    luac.delete_pack %30
    luac.pack_push(%31, [%2])
    %32 = luac.get_fcn_addr %5
    %33 = call_indirect %32(%31) : (!lua.pack) -> !lua.pack
    luac.delete_pack %33
    %c1_i32_11 = constant 1 : i32
    %34 = luac.new_pack[%c1_i32_11]
    luac.pack_push(%34, [%1])
    %35 = luac.get_fcn_addr %5
    %36 = call_indirect %35(%34) : (!lua.pack) -> !lua.pack
    %c3_i32_12 = constant 3 : i32
    %37 = luac.new_pack[%c3_i32_12]
    luac.pack_push(%37, [%0])
    luac.pack_push_all(%37, %36)
    luac.delete_pack %36
    luac.pack_push(%37, [%2])
    %38 = luac.pack_pull_one %37
    %39 = luac.pack_pull_one %37
    %40 = luac.pack_pull_one %37
    %41 = luac.pack_pull_one %37
    luac.delete_pack %37
    %c4_i32 = constant 4 : i32
    %42 = luac.new_pack[%c4_i32]
    luac.pack_push(%42, [%38])
    luac.pack_push(%42, [%39])
    luac.pack_push(%42, [%40])
    luac.pack_push(%42, [%41])
    %43 = luac.get_fcn_addr %5
    %44 = call_indirect %43(%42) : (!lua.pack) -> !lua.pack
    luac.delete_pack %44
    return
  }
}
