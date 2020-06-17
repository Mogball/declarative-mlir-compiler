#include "lua.h"

#include <stddef.h>

TObject lua_add(TObject *lhs, TObject *rhs) {
  return lua_get_nil();
}

TObject lua_sub(TObject *lhs, TObject *rhs) {
  return lua_get_nil();
}

TObject lua_eq(TObject *lhs, TObject *rhs) {
  return lua_get_nil();
}

TObject lua_neq(TObject *lhs, TObject *rhs) {
  return lua_get_nil();
}


TObject lua_get_nil(void) {
  TObject ret;
  ret.type = 0;
  return ret;
}

TObject lua_new_table(void) {
  return lua_get_nil();
}

TObject lua_get_string(const char *data, int32_t len) {
  return lua_get_nil();
}


TObject lua_wrap_int(LuaInteger val) {
  return lua_get_nil();
}

TObject lua_wrap_real(LuaReal val) {
  return lua_get_nil();
}

TObject lua_wrap_bool(bool boolean) {
  return lua_get_nil();
}

LuaInteger lua_unwrap_int(TObject *val) {
  return 0;
}

LuaReal lua_unwrap_real(TObject *val) {
  return 0.0;
}

bool lua_unwrap_bool(TObject *val) {
  return false;
}


TObject lua_typeof(TObject *val) {
  return lua_get_nil();
}


TObject lua_table_get(TObject *tbl, TObject *key) {
  return lua_get_nil();
}

void lua_table_set(TObject *tbl, TObject *key, TObject *val) {
}

TObject lua_table_size(TObject *tbl) {
  return lua_get_nil();
}

void print(TObject *val) {
}

TObject random_string_or_int(size_t len) {
  return lua_get_nil();
}
