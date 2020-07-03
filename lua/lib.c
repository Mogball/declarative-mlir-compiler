#include "lib.h"

#include <stdlib.h>

/*******************************************************************************
 * Library Functions
 ******************************************************************************/

TObject *lua_alloc() {
  return malloc(sizeof(TObject));
}

void lua_alloc_gc(TObject *val) {
  val->gc = malloc(sizeof(TComplex));
}

int16_t lua_get_type(TObject *val) {
  return val->type;
}
void lua_set_type(TObject *val, int16_t ty) {
  val->type = ty;
}

bool lua_get_bool_val(TObject *val) {
  return val->b;
}

void lua_set_bool_val(TObject *val, bool b) {
  val->b = b;
}

int64_t lua_get_int64_val(TObject *val) {
  return val->iv;
}
void lua_set_int64_val(TObject *val, int64_t iv) {
  val->ntype = INT;
  val->iv = iv;
}

double lua_get_double_val(TObject *val) {
  return val->fp;
}
void lua_set_double_val(TObject *val, double fp) {
  val->ntype = REAL;
  val->fp = fp;
}

lua_fcn_t lua_get_fcn_addr(TObject *val) {
  return val->gc->fcn_addr;
}

void lua_set_fcn_addr(TObject *val, lua_fcn_t fcn_addr) {
  val->gc->fcn_addr = fcn_addr;
}

TPack *lua_get_capture_pack(TObject *val) {
  return val->gc->cap_pack;
}

void lua_set_capture_pack(TObject *val, TPack *pack) {
  val->gc->cap_pack = pack;
}

uint64_t lua_get_value_union(TObject *val) {
  return val->u;
}
void lua_set_value_union(TObject *val, uint64_t u) {
  val->u = u;
}

bool lua_is_int(TObject *val) {
  return val->ntype == INT;
}

TPack *lua_new_pack(int64_t rsv) {
  TPack *pack = malloc(sizeof(TPack));
  pack->size = 0;
  pack->idx = 0;
  pack->objs = malloc(sizeof(TObject *) * rsv);
  return pack;
}
void lua_delete_pack(TPack *pack) {
  free(pack->objs);
  free(pack);
}
void lua_pack_push(TPack *pack, TObject *val) {
  pack->objs[pack->size++] = val;
}
TObject *lua_pack_pull_one(TPack *pack) {
  if (pack->idx == pack->size) {
    TObject *ret = lua_alloc();
    lua_set_type(ret, NIL);
    return ret;
  }
  return pack->objs[pack->idx++];
}
void lua_pack_push_all(TPack *pack, TPack *vals) {
  while (vals->idx != vals->size) {
    lua_pack_push(pack, vals->objs[vals->idx++]);
  }
}
int64_t lua_pack_get_size(TPack *pack) {
  return pack->size;
}
void lua_pack_rewind(TPack *pack) {
  pack->idx = 0;
}

extern TObject *lua_table_get_impl(TObject *tbl, TObject *key);
TObject *lua_table_get(TObject *tbl, TObject *key) {
  return lua_table_get_impl(tbl, key);
}

extern TObject *lua_load_string_impl(const char *data, uint64_t len);
TObject *lua_load_string(const char *data, uint64_t len) {
  return lua_load_string_impl(data, len);
}

/*******************************************************************************
 * Builtins
 ******************************************************************************/

extern TObject *builtin_print;
extern TObject *builtin_string;

TObject *lua_builtin_print() {
  return builtin_print;
}
TObject *lua_builtin_string() {
  return builtin_string;
}
