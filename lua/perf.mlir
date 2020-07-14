

module {
  func @lua_main() -> !lua.pack {
    %0 = lua.builtin "print"
    %1 = lua.alloc "ItemCheck"
    %2 = lua.alloc "BottomUpTree"
    %3 = lua.function_def_capture(%2) : !lua.val ["item", "depth"] (%arg0: !lua.val, %arg1: !lua.val) {
      %31 = "luaopt.const_number"() {value = 0.000000e+00 : f64} : () -> !lua.val
      %32 = lua.binary %arg1 ">" %31
      lua.cond_if %32 then () {
        %33 = lua.binary %arg0 "+" %arg0
        %34 = "luaopt.const_number"() {value = 1.000000e+00 : f64} : () -> !lua.val
        %35 = lua.binary %arg1 "-" %34
        lua.copy %arg1 = %35
        %36 = lua.binary %33 "-" %34
        %37 = lua.concat(%36, %arg1) : (!lua.val, !lua.val) -> !lua.pack {operand_segment_sizes = dense<[2, 0]> : vector<2xi64>}
        %38 = lua.call %2(%37)
        %39 = lua.unpack %38 : (!lua.pack) -> !lua.val
        %40 = lua.concat(%33, %arg1) : (!lua.val, !lua.val) -> !lua.pack {operand_segment_sizes = dense<[2, 0]> : vector<2xi64>}
        %41 = lua.call %2(%40)
        %42 = lua.unpack %41 : (!lua.pack) -> !lua.val
        %43 = lua.table
        lua.table_set %43[%34] = %arg0
        %44 = "luaopt.const_number"() {value = 2.000000e+00 : f64} : () -> !lua.val
        lua.table_set %43[%44] = %39
        %45 = "luaopt.const_number"() {value = 3.000000e+00 : f64} : () -> !lua.val
        lua.table_set %43[%45] = %42
        lua.ret(%43) : !lua.val {operand_segment_sizes = dense<[1, 0]> : vector<2xi64>}
      } else () {
        %33 = lua.table
        %34 = "luaopt.const_number"() {value = 1.000000e+00 : f64} : () -> !lua.val
        lua.table_set %33[%34] = %arg0
        lua.ret(%33) : !lua.val {operand_segment_sizes = dense<[1, 0]> : vector<2xi64>}
      }
      lua.ret() :  {operand_segment_sizes = dense<0> : vector<2xi64>}
    }
    lua.copy %2 = %3
    %4 = lua.function_def_capture(%1) : !lua.val ["tree"] (%arg0: !lua.val) {
      %31 = "luaopt.const_number"() {value = 2.000000e+00 : f64} : () -> !lua.val
      %32 = lua.table_get %arg0[%31]
      lua.cond_if %32 then () {
        %33 = "luaopt.const_number"() {value = 1.000000e+00 : f64} : () -> !lua.val
        %34 = lua.table_get %arg0[%33]
        %35 = lua.concat(%32) : (!lua.val) -> !lua.pack {operand_segment_sizes = dense<[1, 0]> : vector<2xi64>}
        %36 = lua.call %1(%35)
        %37 = lua.unpack %36 : (!lua.pack) -> !lua.val
        %38 = lua.binary %34 "+" %37
        %39 = "luaopt.const_number"() {value = 3.000000e+00 : f64} : () -> !lua.val
        %40 = lua.table_get %arg0[%39]
        %41 = lua.concat(%40) : (!lua.val) -> !lua.pack {operand_segment_sizes = dense<[1, 0]> : vector<2xi64>}
        %42 = lua.call %1(%41)
        %43 = lua.unpack %42 : (!lua.pack) -> !lua.val
        %44 = lua.binary %38 "-" %43
        lua.ret(%44) : !lua.val {operand_segment_sizes = dense<[1, 0]> : vector<2xi64>}
      } else () {
        %33 = "luaopt.const_number"() {value = 1.000000e+00 : f64} : () -> !lua.val
        %34 = lua.table_get %arg0[%33]
        lua.ret(%34) : !lua.val {operand_segment_sizes = dense<[1, 0]> : vector<2xi64>}
      }
      lua.ret() :  {operand_segment_sizes = dense<0> : vector<2xi64>}
    }
    lua.copy %1 = %4
    %5 = "luaopt.const_number"() {value = 4.000000e+00 : f64} : () -> !lua.val
    %6 = "luaopt.const_number"() {value = 1.800000e+01 : f64} : () -> !lua.val
    %7 = "luaopt.const_number"() {value = 1.000000e+00 : f64} : () -> !lua.val
    %8 = lua.binary %6 "+" %7
    %9 = "luaopt.const_number"() {value = 0.000000e+00 : f64} : () -> !lua.val
    %10 = lua.concat(%9, %8) : (!lua.val, !lua.val) -> !lua.pack {operand_segment_sizes = dense<[2, 0]> : vector<2xi64>}
    %11 = lua.call %2(%10)
    %12 = lua.unpack %11 : (!lua.pack) -> !lua.val
    %13 = lua.get_string "stretch tree of depth"
    %14 = lua.get_string "check:"
    %15 = lua.concat(%12) : (!lua.val) -> !lua.pack {operand_segment_sizes = dense<[1, 0]> : vector<2xi64>}
    %16 = lua.call %1(%15)
    %17 = lua.concat(%13, %8, %14, %16) : (!lua.val, !lua.val, !lua.val, !lua.pack) -> !lua.pack {operand_segment_sizes = dense<[3, 1]> : vector<2xi64>}
    %18 = lua.call %0(%17)
    %19 = lua.concat(%9, %6) : (!lua.val, !lua.val) -> !lua.pack {operand_segment_sizes = dense<[2, 0]> : vector<2xi64>}
    %20 = lua.call %2(%19)
    %21 = lua.unpack %20 : (!lua.pack) -> !lua.val
    %22 = "luaopt.const_number"() {value = 2.000000e+00 : f64} : () -> !lua.val
    %23 = lua.unary "-" %7
    %24 = lua.get_string "trees of depth"
    lua.numeric_for "depth" in[%5, %6] by %22 do (%arg0: !lua.val) {
      %31 = lua.binary %6 "-" %arg0
      %32 = lua.binary %31 "+" %5
      %33 = lua.binary %22 "^" %32
      %34 = lua.number 0.000000e+00 : f64
      lua.numeric_for "i" in[%7, %33] by %7 do (%arg1: !lua.val) {
        %38 = lua.concat(%7, %arg0) : (!lua.val, !lua.val) -> !lua.pack {operand_segment_sizes = dense<[2, 0]> : vector<2xi64>}
        %39 = lua.call %2(%38)
        %40 = lua.call %1(%39)
        %41 = lua.unpack %40 : (!lua.pack) -> !lua.val
        %42 = lua.binary %34 "+" %41
        %43 = lua.concat(%23, %arg0) : (!lua.val, !lua.val) -> !lua.pack {operand_segment_sizes = dense<[2, 0]> : vector<2xi64>}
        %44 = lua.call %2(%43)
        %45 = lua.call %1(%44)
        %46 = lua.unpack %45 : (!lua.pack) -> !lua.val
        %47 = lua.binary %42 "+" %46
        lua.copy %34 = %47
        lua.end
      }
      %35 = lua.binary %33 "*" %22
      %36 = lua.concat(%35, %24, %arg0, %14, %34) : (!lua.val, !lua.val, !lua.val, !lua.val, !lua.val) -> !lua.pack {operand_segment_sizes = dense<[5, 0]> : vector<2xi64>}
      %37 = lua.call %0(%36)
      lua.end
    }
    %25 = lua.get_string "long lived tree of depth"
    %26 = lua.concat(%21) : (!lua.val) -> !lua.pack {operand_segment_sizes = dense<[1, 0]> : vector<2xi64>}
    %27 = lua.call %1(%26)
    %28 = lua.concat(%25, %6, %14, %27) : (!lua.val, !lua.val, !lua.val, !lua.pack) -> !lua.pack {operand_segment_sizes = dense<[3, 1]> : vector<2xi64>}
    %29 = lua.call %0(%28)
    %30 = lua.concat() : () -> !lua.pack {operand_segment_sizes = dense<0> : vector<2xi64>}
    return %30 : !lua.pack
  }
}
