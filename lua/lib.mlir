module {
  func @luac_check_number_type(%lhs: !lua.ref, %rhs: !lua.ref) -> (!luac.bool, !luac.type_enum) {
    %num_type = constant 2 : !luac.type_enum
    %lhs_type = luac.get_type type(%lhs)
    %rhs_type = luac.get_type type(%rhs)

    %lhs_ok = cmpi "eq", %lhs_type, %num_type : !luac.type_enum
    %rhs_ok = cmpi "eq", %rhs_type, %num_type : !luac.type_enum
    %types_ok = and %lhs_ok, %rhs_ok : !luac.bool

    return %types_ok, %num_type : !luac.bool, !luac.type_enum
  }

  func @luac_get_as_fp(%val: !lua.ref, %is_iv: !luac.bool) -> !luac.real {
    %ret = loop.if %is_iv -> !luac.real {
      %iv = luac.get_int64_val %val
      %fp = sitofp %iv : !luac.integer to !luac.real
      loop.yield %fp : !luac.real
    } else {
      %fp = luac.get_double_val %val
      loop.yield %fp : !luac.real
    }
    return %ret : !luac.real
  }

  func @lua_add(%lhs: !lua.ref, %rhs: !lua.ref) -> !lua.ref {
    %ret = lua.nil

    %types_ok, %num_type = call @luac_check_number_type(%lhs, %rhs)
        : (!lua.ref, !lua.ref) -> (!luac.bool, !luac.type_enum)

    loop.if %types_ok {
      luac.set_type type(%ret) = %num_type
      %lhs_is_iv = luac.is_int %lhs
      %rhs_is_iv = luac.is_int %rhs
      %both_iv = and %lhs_is_iv, %rhs_is_iv : !luac.bool

      loop.if %both_iv {
        %lhs_iv = luac.get_int64_val %lhs
        %rhs_iv = luac.get_int64_val %rhs
        %ret_iv = addi %lhs_iv, %rhs_iv : !luac.integer
        luac.set_int64_val %ret = %ret_iv
      } else {
        %lhs_fp = call @luac_get_as_fp(%lhs, %lhs_is_iv) : (!lua.ref, !luac.bool) -> !luac.real
        %rhs_fp = call @luac_get_as_fp(%rhs, %rhs_is_iv) : (!lua.ref, !luac.bool) -> !luac.real
        %ret_fp = addf %lhs_fp, %rhs_fp : !luac.real
        luac.set_double_val %ret = %ret_fp
      }
    }

    return %ret : !lua.ref
  }

  func @lua_sub(%lhs: !lua.ref, %rhs: !lua.ref) -> !lua.ref {
    %ret = lua.nil

    %types_ok, %num_type = call @luac_check_number_type(%lhs, %rhs)
        : (!lua.ref, !lua.ref) -> (!luac.bool, !luac.type_enum)

    loop.if %types_ok {
      luac.set_type type(%ret) = %num_type
      %lhs_is_iv = luac.is_int %lhs
      %rhs_is_iv = luac.is_int %rhs
      %both_iv = and %lhs_is_iv, %rhs_is_iv : !luac.bool

      loop.if %both_iv {
        %lhs_iv = luac.get_int64_val %lhs
        %rhs_iv = luac.get_int64_val %rhs
        %ret_iv = subi %lhs_iv, %rhs_iv : !luac.integer
        luac.set_int64_val %ret = %ret_iv
      } else {
        %lhs_fp = call @luac_get_as_fp(%lhs, %lhs_is_iv) : (!lua.ref, !luac.bool) -> !luac.real
        %rhs_fp = call @luac_get_as_fp(%rhs, %rhs_is_iv) : (!lua.ref, !luac.bool) -> !luac.real
        %ret_fp = subf %lhs_fp, %rhs_fp : !luac.real
        luac.set_double_val %ret = %ret_fp
      }
    }

    return %ret : !lua.ref
  }

  func @lua_mul(%lhs: !lua.ref, %rhs: !lua.ref) -> !lua.ref {
    %ret = lua.nil

    %types_ok, %num_type = call @luac_check_number_type(%lhs, %rhs)
        : (!lua.ref, !lua.ref) -> (!luac.bool, !luac.type_enum)

    loop.if %types_ok {
      luac.set_type type(%ret) = %num_type
      %lhs_is_iv = luac.is_int %lhs
      %rhs_is_iv = luac.is_int %rhs
      %both_iv = and %lhs_is_iv, %rhs_is_iv : !luac.bool

      loop.if %both_iv {
        %lhs_iv = luac.get_int64_val %lhs
        %rhs_iv = luac.get_int64_val %rhs
        %ret_iv = muli %lhs_iv, %rhs_iv : !luac.integer
        luac.set_int64_val %ret = %ret_iv
      } else {
        %lhs_fp = call @luac_get_as_fp(%lhs, %lhs_is_iv) : (!lua.ref, !luac.bool) -> !luac.real
        %rhs_fp = call @luac_get_as_fp(%rhs, %rhs_is_iv) : (!lua.ref, !luac.bool) -> !luac.real
        %ret_fp = mulf %lhs_fp, %rhs_fp : !luac.real
        luac.set_double_val %ret = %ret_fp
      }
    }

    return %ret : !lua.ref
  }
}
