func @lua_add(%sret: !lua.value, %lhs: !lua.value, %rhs: !lua.value)
func @lua_sub(%sret: !lua.value, %lhs: !lua.value, %rhs: !lua.value)
func @lua_eq(%sret: !lua.value, %lhs: !lua.value, %rhs: !lua.value)
func @lua_neq(%sret: !lua.value, %lhs: !lua.value, %rhs: !lua.value)

func @lua_get_nil(%sret: !lua.value)
func @lua_new_table(%sret: !lua.value)
func @lua_get_string(%sret: !lua.value, %str: !luac.string, %len: i32)

func @lua_wrap_int(%sret: !lua.value, %val: !lua.integer)
func @lua_wrap_real(%sret: !lua.value, %val: !lua.real)
func @lua_wrap_bool(%sret: !lua.value, %val: !lua.bool)
func @lua_unwrap_int(%val: !lua.value) -> !lua.integer
func @lua_unwrap_real(%val: !lua.value) -> !lua.real
func @lua_unwrap_bool(%val: !lua.value) -> !lua.bool

func @lua_typeof(%sret: !lua.value, %val: !lua.value)

func @lua_table_get(%sret: !lua.value, %tbl: !lua.value, %key: !lua.value)
func @lua_table_set(%tbl: !lua.value, %key: !lua.value, %val: !lua.value) -> ()
func @lua_table_size(%sret: !lua.value, %tbl: !lua.value)
