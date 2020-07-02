#include "lib.h"

#include <stdlib.h>

/*******************************************************************************
 * Library Functions
 ******************************************************************************/

TObject *lua_alloc() {
  return malloc(sizeof(TObject));
}

int16_t lua_get_type(TObject *val) {
  return val->type;
}
void lua_set_type(TObject *val, int16_t ty) {
  val->type = ty;
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

/*******************************************************************************
 * Builtins
 ******************************************************************************/

extern TObject *builtin_print;

TObject *lua_builtin_print() {
  return builtin_print;
}