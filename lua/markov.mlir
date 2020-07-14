

module {
  func @lua_main() -> !lua.pack {
    %0 = lua.builtin "print"
    %1 = lua.builtin "math"
    %2 = lua.builtin "table"
    %3 = lua.builtin "string"
    %4 = lua.builtin "io"
    %5 = lua.function_def_capture(%3, %4) : !lua.val, !lua.val [] (%arg0: !lua.val, %arg1: !lua.val) {
      %65 = lua.get_string "read"
      %66 = lua.table_get %arg1[%65]
      %67 = lua.concat() : () -> !lua.pack {operand_segment_sizes = dense<0> : vector<2xi64>}
      %68 = lua.call %66(%67)
      %69 = lua.unpack %68 : (!lua.pack) -> !lua.val
      %70 = "luaopt.const_number"() {value = 1.000000e+00 : f64} : () -> !lua.val
      %71 = lua.function_def_capture(%arg0, %70, %arg1, %69) : !lua.val, !lua.val, !lua.val, !lua.val [] (%arg2: !lua.val, %arg3: !lua.val, %arg4: !lua.val, %arg5: !lua.val) {
        %72 = lua.get_string "find"
        %73 = lua.table_get %arg2[%72]
        %74 = lua.get_string "%w+"
        br ^bb1
      ^bb1:  // 2 preds: ^bb0, ^bb5
        %75 = luac.convert_bool_like %arg5
        cond_br %75, ^bb2, ^bb6
      ^bb2:  // pred: ^bb1
        %76 = lua.concat(%arg5, %74, %arg3) : (!lua.val, !lua.val, !lua.val) -> !lua.pack {operand_segment_sizes = dense<[3, 0]> : vector<2xi64>}
        %77 = lua.call %73(%76)
        %78:2 = lua.unpack %77 : (!lua.pack) -> (!lua.val, !lua.val)
        %79 = luac.convert_bool_like %78#0
        cond_br %79, ^bb3, ^bb4
      ^bb3:  // pred: ^bb2
        %80 = "luaopt.const_number"() {value = 1.000000e+00 : f64} : () -> !lua.val
        %81 = lua.binary %78#1 "+" %80
        lua.copy %arg3 = %81
        %82 = lua.get_string "sub"
        %83 = lua.table_get %arg2[%82]
        %84 = lua.concat(%arg5, %78#0, %78#1) : (!lua.val, !lua.val, !lua.val) -> !lua.pack {operand_segment_sizes = dense<[3, 0]> : vector<2xi64>}
        %85 = lua.call %83(%84)
        lua.ret(%85) : !lua.pack {operand_segment_sizes = dense<[0, 1]> : vector<2xi64>}
      ^bb4:  // pred: ^bb2
        %86 = lua.get_string "read"
        %87 = lua.table_get %arg4[%86]
        %88 = lua.concat() : () -> !lua.pack {operand_segment_sizes = dense<0> : vector<2xi64>}
        %89 = lua.call %87(%88)
        %90 = lua.unpack %89 : (!lua.pack) -> !lua.val
        lua.copy %arg5 = %90
        %91 = "luaopt.const_number"() {value = 1.000000e+00 : f64} : () -> !lua.val
        lua.copy %arg3 = %91
        br ^bb5
      ^bb5:  // pred: ^bb4
        br ^bb1
      ^bb6:  // pred: ^bb1
        %92 = lua.nil
        lua.ret(%92) : !lua.val {operand_segment_sizes = dense<[1, 0]> : vector<2xi64>}
      }
      lua.ret(%71) : !lua.val {operand_segment_sizes = dense<[1, 0]> : vector<2xi64>}
    }
    %6 = lua.function_def_capture() :  ["w1", "w2"] (%arg0: !lua.val, %arg1: !lua.val) {
      %65 = lua.get_string " "
      %66 = lua.binary %65 ".." %arg1
      %67 = lua.binary %arg0 ".." %66
      lua.ret(%67) : !lua.val {operand_segment_sizes = dense<[1, 0]> : vector<2xi64>}
    }
    %7 = lua.alloc "statetab"
    %8 = lua.function_def_capture(%7, %2) : !lua.val, !lua.val ["index", "value"] (%arg0: !lua.val, %arg1: !lua.val, %arg2: !lua.val, %arg3: !lua.val) {
      %65 = lua.table_get %arg0[%arg2]
      %66 = lua.unary "not" %65
      %67 = luac.convert_bool_like %66
      cond_br %67, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      %68 = lua.table
      %69 = lua.get_string "n"
      %70 = "luaopt.const_number"() {value = 0.000000e+00 : f64} : () -> !lua.val
      lua.table_set %68[%69] = %70
      lua.table_set %arg0[%arg2] = %68
      br ^bb3
    ^bb2:  // pred: ^bb0
      br ^bb3
    ^bb3:  // 2 preds: ^bb1, ^bb2
      %71 = lua.get_string "insert"
      %72 = lua.table_get %arg1[%71]
      %73 = lua.concat(%65, %arg3) : (!lua.val, !lua.val) -> !lua.pack {operand_segment_sizes = dense<[2, 0]> : vector<2xi64>}
      %74 = lua.call %72(%73)
      lua.ret() :  {operand_segment_sizes = dense<0> : vector<2xi64>}
    }
    %9 = "luaopt.const_number"() {value = 1.000000e+04 : f64} : () -> !lua.val
    %10 = lua.get_string "\\n"
    %11 = lua.table
    lua.copy %7 = %11
    %12 = lua.alloc "w1"
    %13 = lua.alloc "w2"
    lua.copy %12 = %10
    lua.copy %13 = %10
    %14 = lua.concat() : () -> !lua.pack {operand_segment_sizes = dense<0> : vector<2xi64>}
    %15 = lua.call %5(%14)
    %16:3 = lua.unpack %15 : (!lua.pack) -> (!lua.val, !lua.val, !lua.val)
    %17 = lua.concat(%16#1, %16#2) : (!lua.val, !lua.val) -> !lua.pack {operand_segment_sizes = dense<[2, 0]> : vector<2xi64>}
    %18 = lua.call %16#0(%17)
    %19 = lua.nil
    %20 = lua.unpack %18 : (!lua.pack) -> !lua.val
    br ^bb1
  ^bb1:  // 2 preds: ^bb0, ^bb2
    lua.copy %16#2 = %20
    %21 = luac.ne(%16#2, %19)
    %22 = luac.convert_bool_like %21
    cond_br %22, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %23 = lua.concat(%12, %13) : (!lua.val, !lua.val) -> !lua.pack {operand_segment_sizes = dense<[2, 0]> : vector<2xi64>}
    %24 = lua.call %6(%23)
    %25 = "luaopt.unpack_unsafe"(%24) : (!lua.pack) -> !lua.val
    %26 = lua.concat(%25, %20) : (!lua.val, !lua.val) -> !lua.pack {operand_segment_sizes = dense<[2, 0]> : vector<2xi64>}
    %27 = lua.call %8(%26)
    lua.copy %12 = %13
    lua.copy %13 = %20
    %28 = lua.concat(%16#1, %16#2) : (!lua.val, !lua.val) -> !lua.pack {operand_segment_sizes = dense<[2, 0]> : vector<2xi64>}
    %29 = lua.call %16#0(%28)
    %30 = lua.unpack %29 : (!lua.pack) -> !lua.val
    lua.copy %20 = %30
    br ^bb1
  ^bb3:  // pred: ^bb1
    %31 = lua.concat(%12, %13) : (!lua.val, !lua.val) -> !lua.pack {operand_segment_sizes = dense<[2, 0]> : vector<2xi64>}
    %32 = lua.call %6(%31)
    %33 = "luaopt.unpack_unsafe"(%32) : (!lua.pack) -> !lua.val
    %34 = lua.concat(%33, %10) : (!lua.val, !lua.val) -> !lua.pack {operand_segment_sizes = dense<[2, 0]> : vector<2xi64>}
    %35 = lua.call %8(%34)
    lua.copy %12 = %10
    lua.copy %13 = %10
    %36 = "luaopt.const_number"() {value = 1.000000e+00 : f64} : () -> !lua.val
    %37 = lua.get_string "random"
    %38 = lua.table_get %1[%37]
    %39 = lua.get_string "write"
    %40 = lua.table_get %4[%39]
    %41 = lua.get_string " "
    %42 = luac.get_double_val %36
    %43 = luac.get_double_val %36
    %44 = luac.wrap_real %43
    br ^bb4
  ^bb4:  // 2 preds: ^bb3, ^bb8
    %45 = luac.le(%44, %9)
    %46 = luac.convert_bool_like %45
    cond_br %46, ^bb5, ^bb9
  ^bb5:  // pred: ^bb4
    %47 = lua.concat(%12, %13) : (!lua.val, !lua.val) -> !lua.pack {operand_segment_sizes = dense<[2, 0]> : vector<2xi64>}
    %48 = lua.call %6(%47)
    %49 = "luaopt.unpack_unsafe"(%48) : (!lua.pack) -> !lua.val
    %50 = lua.table_get %7[%49]
    %51 = lua.unary "#" %50
    %52 = lua.concat(%51) : (!lua.val) -> !lua.pack {operand_segment_sizes = dense<[1, 0]> : vector<2xi64>}
    %53 = lua.call %38(%52)
    %54 = lua.unpack %53 : (!lua.pack) -> !lua.val
    %55 = lua.table_get %50[%54]
    %56 = lua.binary %55 "==" %10
    %57 = luac.convert_bool_like %56
    cond_br %57, ^bb6, ^bb7
  ^bb6:  // pred: ^bb5
    %58 = lua.concat() : () -> !lua.pack {operand_segment_sizes = dense<0> : vector<2xi64>}
    %59 = lua.call %0(%58)
    lua.ret() :  {operand_segment_sizes = dense<0> : vector<2xi64>}
  ^bb7:  // pred: ^bb5
    br ^bb8
  ^bb8:  // pred: ^bb7
    %60 = lua.concat(%55, %41) : (!lua.val, !lua.val) -> !lua.pack {operand_segment_sizes = dense<[2, 0]> : vector<2xi64>}
    %61 = lua.call %40(%60)
    lua.copy %12 = %13
    lua.copy %13 = %55
    %62 = luac.get_double_val %44
    %63 = addf %62, %42 : f64
    %64 = luac.get_ref %44
    luac.set_double_val %64 = %63
    br ^bb4
  ^bb9:  // pred: ^bb4
    lua.ret() :  {operand_segment_sizes = dense<0> : vector<2xi64>}
  }
}
