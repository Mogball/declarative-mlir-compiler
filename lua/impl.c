#include "lua.h"

#include <stddef.h>
#include <stdio.h>

void lua_add(TObject *sret, TObject *lhs, TObject *rhs) {
}

void lua_sub(TObject *sret, TObject *lhs, TObject *rhs) {
}

void lua_eq(TObject *sret, TObject *lhs, TObject *rhs) {
}

void lua_neq(TObject *sret, TObject *lhs, TObject *rhs) {
}


void lua_get_nil(TObject *sret) {
}

void lua_new_table(TObject *sret) {
}

void lua_get_string(TObject *sret, const char *data, int32_t len) {
}


void lua_wrap_int(TObject *sret, LuaInteger val) {
}

void lua_wrap_real(TObject *sret, LuaReal val) {
}

void lua_wrap_bool(TObject *sret, bool boolean) {
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


void lua_typeof(TObject *sret, TObject *val) {
}


void lua_table_get(TObject *sret, TObject *tbl, TObject *key) {
}

void lua_table_set(TObject *tbl, TObject *key, TObject *val) {
}

void lua_table_size(TObject *sret, TObject *tbl) {
}

void print(TObject *val) {
}

void random_string_or_int(TObject *sret, size_t len) {
}
