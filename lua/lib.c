#include "lib.h"

#include <stdlib.h>

/*******************************************************************************
 * Library Functions
 ******************************************************************************/

void lua_alloc_gc(TObject *ptr) {
  ptr->gc = malloc(sizeof(TComplex));
}

/*******************************************************************************
 * Getters and setters
 ******************************************************************************/

int32_t lua_get_type(TObject val) {
  return val.type;
}
void lua_set_type(TObject *ptr, int32_t ty) {
  ptr->type = ty;
}

bool lua_get_bool_val(TObject val) {
  return val.b;
}
void lua_set_bool_val(TObject *ptr, bool b) {
  ptr->b = b;
}

double lua_get_double_val(TObject val) {
  return val.num;
}
void lua_set_double_val(TObject *ptr, double num) {
  ptr->num = num;
}

lua_fcn_t lua_get_fcn_addr(TObject val) {
  return val.gc->fcn_addr;
}
void lua_set_fcn_addr(TObject val, lua_fcn_t fcn_addr) {
  val.gc->fcn_addr = fcn_addr;
}

TCapture lua_get_capture_pack(TObject val) {
  return val.gc->capture;
}
void lua_set_capture_pack(TObject val, TCapture capture) {
  val.gc->capture = capture;
}

uint64_t lua_get_value_union(TObject val) {
  return val.u;
}
void lua_set_value_union(TObject *ptr, uint64_t u) {
  ptr->u = u;
}

/*******************************************************************************
 * Capture Packs
 ******************************************************************************/

TCapture lua_new_capture(int32_t size) {
  return malloc(sizeof(TObject *) * size);
}
void lua_add_capture(TCapture capture, TObject *ptr, int32_t idx) {
  capture[idx] = ptr;
}

/*******************************************************************************
 * Value Packs
 ******************************************************************************/

extern TPack *g_ret_pack;
extern TPack *g_arg_pack;

TPack lua_get_ret_pack(int32_t rsv) {
  g_ret_pack->size = rsv;
  g_ret_pack->objs = realloc(g_ret_pack->objs, rsv * sizeof(TObject));
  return *g_ret_pack;
}

TPack lua_get_arg_pack(int32_t rsv) {
  g_arg_pack->size = rsv;
  g_arg_pack->objs = realloc(g_arg_pack->objs, rsv * sizeof(TObject));
  return *g_arg_pack;
}

void lua_pack_insert(TPack pack, TObject val, int32_t idx) {
  pack.objs[idx] = val;
}
void lua_pack_insert_all(TPack pack, TPack tail, int32_t idx) {
  for (int32_t i = 0, e = tail.size; i != e; ++i, ++idx) {
    pack.objs[idx] = tail.objs[i];
  }
}
TObject lua_pack_get(TPack pack, int32_t idx) {
  if (idx >= pack.size) {
    TObject ret;
    ret.type = NIL;
    return ret;
  }
  return pack.objs[idx];
}
int32_t lua_pack_get_size(TPack pack) {
  return pack.size;
}

/*******************************************************************************
 * Tables
 ******************************************************************************/

extern void lua_init_table_impl(TObject tbl);
extern void lua_table_set_impl(TObject tbl, TObject key, TObject val);
extern TObject lua_table_get_impl(TObject tbl, TObject key);
void lua_init_table(TObject tbl) {
  lua_init_table_impl(tbl);
}
void lua_table_set(TObject tbl, TObject key, TObject val) {
  lua_table_set_impl(tbl, key, val);
}
TObject lua_table_get(TObject tbl, TObject key) {
  return lua_table_get_impl(tbl, key);
}

extern void lua_table_set_prealloc_impl(TObject tbl, int64_t iv, TObject val);
extern TObject lua_table_get_prealloc_impl(TObject tbl, int64_t iv);
void lua_table_set_prealloc(TObject tbl, int64_t iv, TObject val) {
  lua_table_set_prealloc_impl(tbl, iv, val);
}
TObject lua_table_get_prealloc(TObject tbl, int64_t iv) {
  return lua_table_get_prealloc_impl(tbl, iv);
}

/*******************************************************************************
 * Strings
 ******************************************************************************/

extern TObject lua_load_string_impl(const char *data, uint64_t len);
TObject lua_load_string(const char *data, uint64_t len) {
  return lua_load_string_impl(data, len);
}

/*******************************************************************************
 * Builtins
 ******************************************************************************/

extern TObject *builtin_print;
extern TObject *builtin_string;
extern TObject *builtin_table;
extern TObject *builtin_io;
extern TObject *builtin_math;

TObject lua_builtin_print(void) {
  return *builtin_print;
}
TObject lua_builtin_string(void) {
  return *builtin_string;
}
TObject lua_builtin_table(void) {
  return *builtin_table;
}
TObject lua_builtin_io(void) {
  return *builtin_io;
}
TObject lua_builtin_math(void) {
  return *builtin_math;
}
