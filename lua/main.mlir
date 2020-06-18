func @add_integers_in(%tbl : !lua.value) -> !lua.value {
  %size_v = lua.table_size %tbl
  %size = lua.unwrap %size_v -> !lua.integer

  %acc_value = constant 0 : !lua.integer
  %acc = lua.wrap %acc_value : !lua.integer

  %i0 = constant 0 : index
  %step = constant 1 : index
  %end = index_cast %size : !lua.integer to index

  %ret = loop.for %i = %i0 to %end step %step iter_args(%acc_it = %acc) -> !lua.value {
    %key_v = index_cast %i : index to !lua.integer
    %key = lua.wrap %key_v : !lua.integer

    %val = lua.table_get %tbl[%key]

    %val_type = lua.typeof %val
    %target_type = lua.get_string "number"
    %are_eq = lua.eq(%val_type, %target_type)
    %are_eq_v = lua.unwrap %are_eq -> !lua.bool

    %acc_next = loop.if %are_eq_v -> !lua.value {
      %new_acc = lua.add(%acc_it, %val)
      loop.yield %new_acc : !lua.value
    } else {
      loop.yield %acc_it : !lua.value
    }
    loop.yield %acc_next : !lua.value
  }

  return %ret : !lua.value
}

func @random_string_or_int(%sret: !lua.value, %len : index)
func @print(%val : !lua.value)

func @main() -> () {
  %tbl = lua.new_table

  %L = constant 16 : index

  %i0 = constant 0 : index
  %in = constant 10 : index
  %step = constant 1 : index
  loop.for %i = %i0 to %in step %step {
    %key_v = index_cast %i : index to !lua.integer
    %key = lua.wrap %key_v : !lua.integer
    %val = lualib.alloc
    call @random_string_or_int(%val, %L) : (!lua.value, index) -> ()
    lua.table_set %tbl[%key] = %val
  }

  call @print(%tbl) : (!lua.value) -> ()
  %result = call @add_integers_in(%tbl) : (!lua.value) -> !lua.value
  call @print(%result) : (!lua.value) -> ()

  return
}
