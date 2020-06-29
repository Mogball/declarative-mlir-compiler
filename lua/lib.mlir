module {
  func @lua_add(%lhs: !lua.ref, %rhs: !lua.ref) -> !lua.ref {
    %ret = luac.alloc
    %nil_type = constant 0 : i32
    luac.set_type type(%ret) = %nil_type

    %num_type = constant 2 : i32
    %lhs_type = luac.get_type type(%lhs)
    %rhs_type = luac.get_type type(%rhs)

    %lhs_ok = cmpi "eq", %lhs_type, %num_type : i32
    %rhs_ok = cmpi "eq", %rhs_type, %num_type : i32
    %types_ok = and %lhs_ok, %rhs_ok : i1

    loop.if %types_ok {
      luac.set_type type(%ret) = %num_type
      %lhs_is_fp = luac.is_float %lhs
      %rhs_is_fp = luac.is_float %rhs
      loop.if %lhs_is_fp {
        %lhs_fp = luac.get_double_val %lhs
        loop.if %rhs_is_fp {
          %rhs_fp = luac.get_double_val %rhs
          %ret_fp = addf %lhs_fp, %rhs_fp : f64
          luac.set_double_val %ret = %ret_fp
        } else {
          %rhs_iv = luac.get_int64_val %rhs
          %rhs_fp = sitofp %rhs_iv : i64 to f64
          %ret_fp = addf %lhs_fp, %rhs_fp : f64
          luac.set_double_val %ret = %ret_fp
        }
      } else {
        %lhs_iv = luac.get_int64_val %lhs
        loop.if %rhs_is_fp {
          %rhs_fp = luac.get_double_val %rhs
          %lhs_fp = sitofp %lhs_iv : i64 to f64
          %ret_fp = addf %lhs_fp, %rhs_fp : f64
          luac.set_double_val %ret = %ret_fp
        } else {
          %rhs_iv = luac.get_int64_val %rhs
          %ret_iv = addi %lhs_iv, %rhs_iv : i64
          luac.set_int64_val %ret = %ret_iv
        }
      }
    }

    return %ret : !lua.ref
  }
}
