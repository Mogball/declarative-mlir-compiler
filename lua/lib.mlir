module {
  // debugging
  func @print_one(%val: !lua.val) -> ()

  func @luac_check_number_type(%lhs: !lua.val, %rhs: !lua.val) -> i1 {
    %num_type = constant #luac.type_num
    %lhs_type = luac.get_type type(%lhs)
    %rhs_type = luac.get_type type(%rhs)

    %lhs_ok = cmpi "eq", %lhs_type, %num_type : !luac.type_enum
    %rhs_ok = cmpi "eq", %rhs_type, %num_type : !luac.type_enum
    %types_ok = and %lhs_ok, %rhs_ok : i1

    return %types_ok : i1
  }

  func @lua_add(%lhs: !lua.val, %rhs: !lua.val) -> !lua.val {
    %types_ok = call @luac_check_number_type(%lhs, %rhs)
        : (!lua.val, !lua.val) -> i1
    %ret = scf.if %types_ok -> !lua.val {
      %lhs_num = luac.get_double_val %lhs
      %rhs_num = luac.get_double_val %rhs
      %ret_num = addf %lhs_num, %rhs_num : !luac.real
      %ret_v = luac.wrap_real %ret_num
      scf.yield %ret_v : !lua.val
    } else {
      %ret_v = lua.nil
      scf.yield %ret_v : !lua.val
    }
    return %ret : !lua.val
  }

  func @lua_sub(%lhs: !lua.val, %rhs: !lua.val) -> !lua.val {
    %types_ok = call @luac_check_number_type(%lhs, %rhs)
        : (!lua.val, !lua.val) -> i1
    %ret = scf.if %types_ok -> !lua.val {
      %lhs_num = luac.get_double_val %lhs
      %rhs_num = luac.get_double_val %rhs
      %ret_num = subf %lhs_num, %rhs_num : !luac.real
      %ret_v = luac.wrap_real %ret_num
      scf.yield %ret_v : !lua.val
    } else {
      %ret_v = lua.nil
      scf.yield %ret_v : !lua.val
    }
    return %ret : !lua.val
  }

  func @lua_mul(%lhs: !lua.val, %rhs: !lua.val) -> !lua.val {
    %types_ok = call @luac_check_number_type(%lhs, %rhs)
        : (!lua.val, !lua.val) -> i1
    %ret = scf.if %types_ok -> !lua.val {
      %lhs_num = luac.get_double_val %lhs
      %rhs_num = luac.get_double_val %rhs
      %ret_num = mulf %lhs_num, %rhs_num : !luac.real
      %ret_v = luac.wrap_real %ret_num
      scf.yield %ret_v : !lua.val
    } else {
      %ret_v = lua.nil
      scf.yield %ret_v : !lua.val
    }
    return %ret : !lua.val
  }

  // double pow(double x, double y) from <math.h>
  func @pow(%x: f64, %y: f64) -> f64
  func @lua_pow(%lhs: !lua.val, %rhs: !lua.val) -> !lua.val {
    %types_ok = call @luac_check_number_type(%lhs, %rhs)
        : (!lua.val, !lua.val) -> i1
    %ret = scf.if %types_ok -> !lua.val {
      %lhs_num = luac.get_double_val %lhs
      %rhs_num = luac.get_double_val %rhs
      %ret_num = call @pow(%lhs_num, %rhs_num) : (f64, f64) -> f64
      %ret_v = luac.wrap_real %ret_num
      scf.yield %ret_v : !lua.val
    } else {
      %ret_v = lua.nil
      scf.yield %ret_v : !lua.val
    }
    return %ret : !lua.val
  }

  func @lua_neg(%val: !lua.val) -> !lua.val {
    %num_type = constant #luac.type_num
    %val_type = luac.get_type type(%val)
    %type_ok = cmpi "eq", %num_type, %val_type : !luac.type_enum
    %ret = scf.if %type_ok -> !lua.val {
      %inv = constant -1.0 : !luac.real
      %num = luac.get_double_val %val
      %ret_num = mulf %inv, %num : !luac.real
      %ret_v = luac.wrap_real %ret_num
      scf.yield %ret_v : !lua.val
    } else {
      %ret_v = lua.nil
      scf.yield %ret_v : !lua.val
    }
    return %ret : !lua.val
  }

  func @lua_lt(%lhs: !lua.val, %rhs: !lua.val) -> !lua.val {
    %types_ok = call @luac_check_number_type(%lhs, %rhs)
        : (!lua.val, !lua.val) -> i1
    %ret = scf.if %types_ok -> !lua.val {
      %lhs_num = luac.get_double_val %lhs
      %rhs_num = luac.get_double_val %rhs
      %ret_b = cmpf "olt", %lhs_num, %rhs_num : !luac.real
      %ret_v = luac.wrap_bool %ret_b
      scf.yield %ret_v : !lua.val
    } else {
      %ret_v = lua.nil
      scf.yield %ret_v : !lua.val
    }
    return %ret : !lua.val
  }

  func @lua_gt(%lhs: !lua.val, %rhs: !lua.val) -> !lua.val {
    %types_ok = call @luac_check_number_type(%lhs, %rhs)
        : (!lua.val, !lua.val) -> i1
    %ret = scf.if %types_ok -> !lua.val {
      %lhs_num = luac.get_double_val %lhs
      %rhs_num = luac.get_double_val %rhs
      %ret_b = cmpf "ogt", %lhs_num, %rhs_num : !luac.real
      %ret_v = luac.wrap_bool %ret_b
      scf.yield %ret_v : !lua.val
    } else {
      %ret_v = lua.nil
      scf.yield %ret_v : !lua.val
    }
    return %ret : !lua.val
  }

  func @lua_le(%lhs: !lua.val, %rhs: !lua.val) -> !lua.val {
    %types_ok = call @luac_check_number_type(%lhs, %rhs)
        : (!lua.val, !lua.val) -> i1
    %ret = scf.if %types_ok -> !lua.val {
      %lhs_num = luac.get_double_val %lhs
      %rhs_num = luac.get_double_val %rhs
      %ret_b = cmpf "ole", %lhs_num, %rhs_num : !luac.real
      %ret_v = luac.wrap_bool %ret_b
      scf.yield %ret_v : !lua.val
    } else {
      %ret_v = lua.nil
      scf.yield %ret_v : !lua.val
    }
    return %ret : !lua.val
  }

  func @lua_bool_and(%lhs: !lua.val, %rhs: !lua.val) -> !lua.val {
    %lhs_b = call @lua_convert_bool_like(%lhs) : (!lua.val) -> i1
    %rhs_b = call @lua_convert_bool_like(%rhs) : (!lua.val) -> i1

    %ret_b = and %lhs_b, %rhs_b : i1
    %ret = luac.wrap_bool %ret_b
    return %ret : !lua.val
  }

  func @lua_bool_not(%val: !lua.val) -> !lua.val {
    %b = call @lua_convert_bool_like(%val) : (!lua.val) -> i1
    %const1 = constant 1 : i1

    %not_b = xor %b, %const1 : i1
    %ret = luac.wrap_bool %not_b
    return %ret : !lua.val
  }

  func @lua_list_size_impl(%impl: !luac.void_ptr) -> i64
  func @lua_list_size(%val: !lua.val) -> !lua.val {
    %table_type = constant #luac.type_tbl
    %type = luac.get_type type(%val)
    %type_ok = cmpi "eq", %type, %table_type : !luac.type_enum

    %ret = scf.if %type_ok -> !lua.val {
      %impl = luac.get_impl %val
      %sz = call @lua_list_size_impl(%impl) : (!luac.void_ptr) -> i64
      %ret_num = sitofp %sz : i64 to !luac.real
      %ret_v = luac.wrap_real %ret_num
      scf.yield %ret_v : !lua.val
    } else {
      %ret_v = lua.nil
      scf.yield %ret_v : !lua.val
    }
    return %ret : !lua.val
  }

  func @lua_strcat_impl(%lhs: !luac.void_ptr,
                        %rhs: !luac.void_ptr) -> !lua.val
  func @lua_strcat(%lhs: !lua.val, %rhs: !lua.val) -> !lua.val {
    %lhs_type = luac.get_type type(%lhs)
    %rhs_type = luac.get_type type(%rhs)
    %str_type = constant #luac.type_str
    %lhs_ok = cmpi "eq", %lhs_type, %str_type : !luac.type_enum
    %rhs_ok = cmpi "eq", %rhs_type, %str_type : !luac.type_enum
    %types_ok = and %lhs_ok, %rhs_ok : i1

    %ret = scf.if %types_ok -> !lua.val {
      %lhs_impl = luac.get_impl %lhs
      %rhs_impl = luac.get_impl %rhs
      %ret_v = call @lua_strcat_impl(%lhs_impl, %rhs_impl)
          : (!luac.void_ptr, !luac.void_ptr) -> !lua.val
      scf.yield %ret_v : !lua.val
    } else {
      %ret_v = lua.nil
      scf.yield %ret_v : !lua.val
    }
    return %ret : !lua.val
  }

  func @lua_eq_impl(%lhs: !lua.val, %rhs: !lua.val) -> i1
  func @lua_eq(%lhs: !lua.val, %rhs: !lua.val) -> !lua.val {
    %lhs_type = luac.get_type type(%lhs)
    %rhs_type = luac.get_type type(%rhs)
    %same_type = cmpi "eq", %lhs_type, %rhs_type : !luac.type_enum

    %ret_b = scf.if %same_type -> i1 {
      %result = call @lua_eq_impl(%lhs, %rhs)
          : (!lua.val, !lua.val) -> i1
      scf.yield %result : i1
    } else {
      %false = constant 0 : i1
      scf.yield %false : i1
    }
    %ret = luac.wrap_bool %ret_b
    return %ret : !lua.val
  }

  func @lua_ne(%lhs: !lua.val, %rhs: !lua.val) -> !lua.val {
    %are_eq = call @lua_eq(%lhs, %rhs) : (!lua.val, !lua.val) -> !lua.val
    %are_eq_b = luac.get_bool_val %are_eq
    %const1 = constant 1 : i1
    %are_ne_b = xor %are_eq_b, %const1 : i1

    %ret = luac.wrap_bool %are_ne_b
    return %ret : !lua.val
  }

  func @lua_convert_bool_like(%val: !lua.val) -> i1 {
    %type = luac.get_type type(%val)

    %nil_type = constant #luac.type_nil
    %is_nil = cmpi "eq", %type, %nil_type : !luac.type_enum

    %ret_b = scf.if %is_nil -> i1 {
      %false = constant 0 : i1
      scf.yield %false : i1
    } else {
      %bool_type = constant #luac.type_bool
      %is_bool = cmpi "eq", %type, %bool_type : !luac.type_enum
      %ret = scf.if %is_bool -> i1 {
        %b = luac.get_bool_val %val
        scf.yield %b : i1
      } else {
        %true = constant 1 : i1
        scf.yield %true : i1
      }
      scf.yield %ret : i1
    }

    return %ret_b : i1
  }

  func @lua_pack_get(%pack: !lua.pack, %idx: i32) -> !lua.val {
    %sz = luac.pack_get_size %pack
    %inside = cmpi "slt", %idx, %sz : i32
    %ret = scf.if %inside -> !lua.val {
      %ret_v = luac.pack_get_unsafe %pack[%idx]
      scf.yield %ret_v : !lua.val
    } else {
      %ret_v = lua.nil
      scf.yield %ret_v : !lua.val
    }
    return %ret : !lua.val
  }
}
