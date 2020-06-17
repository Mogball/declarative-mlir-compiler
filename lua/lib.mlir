func @lua_add(%lhs: !lua.value, %rhs: !lua.value) -> !lua.value
func @lua_sub(%lhs: !lua.value, %rhs: !lua.value) -> !lua.value
func @lua_eq(%lhs: !lua.value, %rhs: !lua.value) -> !lua.bool
func @lua_neq(%lhs: !lua.value, %rhs: !lua.value) -> !lua.bool

func @lua_get_nil() -> !lua.value
func @lua_new_table() -> !lua.value
func @lua_get_string(%str: !luac.string, %len: i32) -> !lua.value

func @lua_wrap_int(%val: !lua.integer) -> !lua.value
func @lua_wrap_real(%val: !lua.real) -> !lua.value
func @lua_wrap_bool(%val: !lua.bool) -> !lua.value
func @lua_unwrap_int(%val: !lua.value) -> !lua.integer
func @lua_unwrap_real(%val: !lua.value) -> !lua.real
func @lua_unwrap_bool(%val: !lua.value) -> !lua.bool

func @lua_typeof(%val: !lua.value) -> !lua.value

func @lua_table_get(%tbl: !lua.value, %key: !lua.value) -> !lua.value
func @lua_table_set(%tbl: !lua.value, %key: !lua.value, %val: !lua.value) -> ()
func @lua_table_size(%tbl: !lua.value) -> !lua.value
