func @add_integers_in(%arg : !lua.value) -> !lua.value {
  %tbl = lua.convto %arg -> !lua.table
  %acc = constant 0 : !lua.integer

  %i0 = constant 0 : index
  %step = constant 1 : index
  %size = lua.size(%tbl)
  %end = index_cast %size : !lua.integer to index

  %ret = loop.for %i = %i0 to %end step %step iter_args(%sum_it = %acc) -> !lua.integer {
    %key = index_cast %i : index to !lua.integer
    %key_v = lua.tovalue(%key : !lua.integer)

    %val = lua.get %tbl[%key_v]

    %val_type = lua.typeof(%val)
    %target_type = lua.string_const "integer"
    %val_type_v = lua.tovalue(%val_type : !lua.string)
    %target_type_v = lua.tovalue(%target_type : !lua.string)
    %are_eq = lua.eq(%val_type_v, %target_type_v)

    %acc_next = loop.if %are_eq -> !lua.integer {
      %acc_v = lua.tovalue(%sum_it : !lua.integer)
      %new_acc_v = lua.add(%acc_v, %val)
      %new_acc = lua.convto %new_acc_v -> !lua.integer
      loop.yield %new_acc : !lua.integer
    } else {
      loop.yield %sum_it : !lua.integer
    }
    loop.yield %acc_next : !lua.integer
  }

  %ret_v = lua.tovalue(%ret : !lua.integer)
  return %ret_v : !lua.value
}

func @lua_main() -> () {
  return
}
