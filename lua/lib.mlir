module {
  // debugging
  func @print_one(%val: !lua.ref) -> ()

  func @luac_check_number_type(%lhs: !lua.ref, %rhs: !lua.ref) -> (!luac.bool, !luac.type_enum) {
    %num_type = constant #luac.type_num
    %lhs_type = luac.get_type type(%lhs)
    %rhs_type = luac.get_type type(%rhs)

    %lhs_ok = cmpi "eq", %lhs_type, %num_type : !luac.type_enum
    %rhs_ok = cmpi "eq", %rhs_type, %num_type : !luac.type_enum
    %types_ok = and %lhs_ok, %rhs_ok : !luac.bool

    return %types_ok, %num_type : !luac.bool, !luac.type_enum
  }

  func @lua_add(%lhs: !lua.ref, %rhs: !lua.ref) -> !lua.ref {
    %ret = lua.nil
    %types_ok, %num_type = call @luac_check_number_type(%lhs, %rhs)
        : (!lua.ref, !lua.ref) -> (!luac.bool, !luac.type_enum)
    loop.if %types_ok {
      luac.set_type type(%ret) = %num_type
      %lhs_num = luac.get_double_val %lhs
      %rhs_num = luac.get_double_val %rhs
      %ret_num = addf %lhs_num, %rhs_num : !luac.real
      luac.set_double_val %ret = %ret_num
    }
    return %ret : !lua.ref
  }

  func @lua_sub(%lhs: !lua.ref, %rhs: !lua.ref) -> !lua.ref {
    %ret = lua.nil
    %types_ok, %num_type = call @luac_check_number_type(%lhs, %rhs)
        : (!lua.ref, !lua.ref) -> (!luac.bool, !luac.type_enum)
    loop.if %types_ok {
      luac.set_type type(%ret) = %num_type
      %lhs_num = luac.get_double_val %lhs
      %rhs_num = luac.get_double_val %rhs
      %ret_num = subf %lhs_num, %rhs_num : !luac.real
      luac.set_double_val %ret = %ret_num
    }
    return %ret : !lua.ref
  }

  func @lua_mul(%lhs: !lua.ref, %rhs: !lua.ref) -> !lua.ref {
    %ret = lua.nil
    %types_ok, %num_type = call @luac_check_number_type(%lhs, %rhs)
        : (!lua.ref, !lua.ref) -> (!luac.bool, !luac.type_enum)
    loop.if %types_ok {
      luac.set_type type(%ret) = %num_type
      %lhs_num = luac.get_double_val %lhs
      %rhs_num = luac.get_double_val %rhs
      %ret_num = mulf %lhs_num, %rhs_num : !luac.real
      luac.set_double_val %ret = %ret_num
    }
    return %ret : !lua.ref
  }

  // double pow(double x, double y) from <math.h>
  func @pow(%x: f64, %y: f64) -> f64
  func @lua_pow(%lhs: !lua.ref, %rhs: !lua.ref) -> !lua.ref {
    %ret = lua.nil
    %types_ok, %num_type = call @luac_check_number_type(%lhs, %rhs)
        : (!lua.ref, !lua.ref) -> (!luac.bool, !luac.type_enum)
    loop.if %types_ok {
      luac.set_type type(%ret) = %num_type
      %lhs_num = luac.get_double_val %lhs
      %rhs_num = luac.get_double_val %rhs
      %ret_num = call @pow(%lhs_num, %rhs_num) : (f64, f64) -> f64
      luac.set_double_val %ret = %ret_num
    }
    return %ret : !lua.ref
  }

  func @lua_neg(%val: !lua.ref) -> !lua.ref {
    %ret = lua.nil
    %num_type = constant #luac.type_num
    %val_type = luac.get_type type(%val)
    %type_ok = cmpi "eq", %num_type, %val_type : !luac.type_enum
    loop.if %type_ok {
      %inv = constant -1.0 : !luac.real
      %num = luac.get_double_val %val
      %ret_num = mulf %inv, %num : !luac.real
      luac.set_double_val %ret = %ret_num
    }
    return %ret : !lua.ref
  }

  func @lua_lt(%lhs: !lua.ref, %rhs: !lua.ref) -> !lua.ref {
    %ret = lua.nil
    %types_ok, %num_type = call @luac_check_number_type(%lhs, %rhs)
        : (!lua.ref, !lua.ref) -> (!luac.bool, !luac.type_enum)
    loop.if %types_ok {
      %bool_type = constant #luac.type_bool
      luac.set_type type(%ret) = %bool_type
      %lhs_num = luac.get_double_val %lhs
      %rhs_num = luac.get_double_val %rhs
      %ret_b = cmpf "olt", %lhs_num, %rhs_num : !luac.real
      luac.set_bool_val %ret = %ret_b
    }
    return %ret : !lua.ref
  }

  func @lua_gt(%lhs: !lua.ref, %rhs: !lua.ref) -> !lua.ref {
    %ret = lua.nil
    %types_ok, %num_type = call @luac_check_number_type(%lhs, %rhs)
        : (!lua.ref, !lua.ref) -> (!luac.bool, !luac.type_enum)
    loop.if %types_ok {
      %bool_type = constant #luac.type_bool
      luac.set_type type(%ret) = %bool_type
      %lhs_num = luac.get_double_val %lhs
      %rhs_num = luac.get_double_val %rhs
      %ret_b = cmpf "ogt", %lhs_num, %rhs_num : !luac.real
      luac.set_bool_val %ret = %ret_b
    }
    return %ret : !lua.ref
  }

  func @lua_le(%lhs: !lua.ref, %rhs: !lua.ref) -> !lua.ref {
    %ret = lua.nil
    %types_ok, %num_type = call @luac_check_number_type(%lhs, %rhs)
        : (!lua.ref, !lua.ref) -> (!luac.bool, !luac.type_enum)
    loop.if %types_ok {
      %bool_type = constant #luac.type_bool
      luac.set_type type(%ret) = %bool_type
      %lhs_num = luac.get_double_val %lhs
      %rhs_num = luac.get_double_val %rhs
      %ret_b = cmpf "ole", %lhs_num, %rhs_num : !luac.real
      luac.set_bool_val %ret = %ret_b
    }
    return %ret : !lua.ref
  }

  func @lua_bool_and(%lhs: !lua.ref, %rhs: !lua.ref) -> !lua.ref {
    %lhs_b = call @lua_convert_bool_like(%lhs) : (!lua.ref) -> !luac.bool
    %rhs_b = call @lua_convert_bool_like(%rhs) : (!lua.ref) -> !luac.bool

    %ret = luac.alloc
    %bool_type = constant #luac.type_bool
    luac.set_type type(%ret) = %bool_type
    %ret_b = and %lhs_b, %rhs_b : !luac.bool
    luac.set_bool_val %ret = %ret_b

    return %ret : !lua.ref
  }

  func @lua_bool_not(%val: !lua.ref) -> !lua.ref {
    %b = call @lua_convert_bool_like(%val) : (!lua.ref) -> !luac.bool
    %const1 = constant 1 : !luac.bool
    %not_b = xor %b, %const1 : !luac.bool

    %ret = luac.alloc
    %bool_type = constant #luac.type_bool
    luac.set_type type(%ret) = %bool_type
    luac.set_bool_val %ret = %not_b

    return %ret : !lua.ref
  }

  func @lua_list_size_impl(%val: !lua.ref) -> i64
  func @lua_list_size(%val: !lua.ref) -> !lua.ref {
    %ret = lua.nil

    %table_type = constant #luac.type_tbl
    %type = luac.get_type type(%val)
    %type_ok = cmpi "eq", %type, %table_type : !luac.type_enum

    loop.if %type_ok {
      %sz = call @lua_list_size_impl(%val) : (!lua.ref) -> i64
      %ret_num = sitofp %sz : i64 to !luac.real
      %num_type = constant #luac.type_num
      luac.set_type type(%ret) = %num_type
      luac.set_double_val %ret = %ret_num
    }

    return %ret : !lua.ref
  }

  func @lua_strcat_impl(%dest: !lua.ref, %lhs: !lua.ref, %rhs: !lua.ref) -> ()
  func @lua_strcat(%lhs: !lua.ref, %rhs: !lua.ref) -> !lua.ref {
    %ret = lua.nil

    %lhs_type = luac.get_type type(%lhs)
    %rhs_type = luac.get_type type(%rhs)
    %str_type = constant #luac.type_str
    %lhs_ok = cmpi "eq", %lhs_type, %str_type : !luac.type_enum
    %rhs_ok = cmpi "eq", %rhs_type, %str_type : !luac.type_enum
    %types_ok = and %lhs_ok, %rhs_ok : !luac.bool

    loop.if %types_ok {
      luac.set_type type(%ret) = %str_type
      luac.alloc_gc %ret
      call @lua_strcat_impl(%ret, %lhs, %rhs) : (!lua.ref, !lua.ref, !lua.ref) -> ()
    }

    return %ret : !lua.ref
  }

  func @lua_eq_impl(%lhs: !lua.ref, %rhs: !lua.ref) -> !luac.bool
  func @lua_eq(%lhs: !lua.ref, %rhs: !lua.ref) -> !lua.ref {
    %lhs_type = luac.get_type type(%lhs)
    %rhs_type = luac.get_type type(%rhs)
    %same_type = cmpi "eq", %lhs_type, %rhs_type : !luac.type_enum

    %ret_bool = loop.if %same_type -> !luac.bool {
      %result = call @lua_eq_impl(%lhs, %rhs) : (!lua.ref, !lua.ref) -> !luac.bool
      loop.yield %result : !luac.bool
    } else {
      %false = constant 0 : !luac.bool
      loop.yield %false : !luac.bool
    }
    %ret = luac.alloc
    %bool_type = constant #luac.type_bool
    luac.set_type type(%ret) = %bool_type
    luac.set_bool_val %ret = %ret_bool
    return %ret : !lua.ref
  }

  func @lua_ne(%lhs: !lua.ref, %rhs: !lua.ref) -> !lua.ref {
    %are_eq = call @lua_eq(%lhs, %rhs) : (!lua.ref, !lua.ref) -> !lua.ref
    %are_eq_b = luac.get_bool_val %are_eq
    %const1 = constant 1 : !luac.bool
    %are_ne_b = xor %are_eq_b, %const1 : !luac.bool

    %ret = luac.alloc
    %bool_type = constant #luac.type_bool
    luac.set_type type(%ret) = %bool_type
    luac.set_bool_val %ret = %are_ne_b
    return %ret : !lua.ref
  }

  func @lua_convert_bool_like(%val: !lua.ref) -> !luac.bool {
    %nil_type = constant #luac.type_nil
    %bool_type = constant #luac.type_bool

    %type = luac.get_type type(%val)
    %is_nil = cmpi "eq", %type, %nil_type : !luac.type_enum

    %ret_b = loop.if %is_nil -> !luac.bool {
      %false = constant 0 : !luac.bool
      loop.yield %false : !luac.bool
    } else {
      %is_bool = cmpi "eq", %type, %bool_type : !luac.type_enum
      %ret = loop.if %is_bool -> !luac.bool {
        %b = luac.get_bool_val %val
        loop.yield %b : !luac.bool
      } else {
        %true = constant 1 : !luac.bool
        loop.yield %true : !luac.bool
      }
      loop.yield %ret : !luac.bool
    }

    return %ret_b : !luac.bool
  }
}
