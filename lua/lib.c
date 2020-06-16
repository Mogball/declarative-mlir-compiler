#include "lua.h"

void hold() {
#define HOLD(fcn, ...) (void *) fcn
  HOLD(lua_add);
  HOLD(lua_sub);
  HOLD(lua_eq);
  HOLD(lua_neq);

  HOLD(lua_get_nil);
  HOLD(lua_new_table);
  HOLD(lua_get_string);

  HOLD(lua_wrap_int);
  HOLD(lua_wrap_real);
  HOLD(lua_wrap_bool);
  HOLD(lua_unwrap_int);
  HOLD(lua_unwrap_real);
  HOLD(lua_unwrap_bool);

  HOLD(lua_typeof);

  HOLD(lua_table_get);
  HOLD(lua_table_set);
  HOLD(lua_table_size);
}
