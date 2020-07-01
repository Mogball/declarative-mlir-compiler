#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************
 * Definitions
 ******************************************************************************/

enum {
  INT,
  REAL
};

enum {
  NIL,
  BOOL,
  NUM,
  STR,
  TBL,
  FCN
};

struct Object;

typedef struct Pack {
  int64_t size;
  int64_t idx;
  struct Object **objs;
} TPack;

typedef TPack *(*lua_fcn_t)(TPack *);

typedef union Complex {
  lua_fcn_t fcn_addr;
} TComplex;

typedef struct Object {
  int16_t type;
  int16_t ntype;
  union {
    uint64_t u;

    bool b;
    int64_t iv;
    double fp;
    TComplex *gc;
  };
} TObject;

TObject *lua_alloc();
int16_t lua_get_type(TObject *val);
void lua_set_type(TObject *val, int16_t ty);
int64_t lua_get_int64_val(TObject *val);
void lua_set_int64_val(TObject *val, int64_t iv);
double lua_get_double_val(TObject *val);
void lua_set_double_val(TObject *val, double fp);
lua_fcn_t lua_get_fcn_addr(TObject *val);
uint64_t lua_get_value_union(TObject *val);
void lua_set_value_union(TObject *val, uint64_t u);
bool lua_is_int(TObject *val);
TPack *lua_new_pack(int64_t size);
void lua_delete_pack(TPack *pack);
void lua_pack_push(TPack *pack, TObject *val);
TObject *lua_pack_pull_one(TPack *pack);
void lua_pack_push_all(TPack *pack, TPack *vals);
int64_t lua_pack_get_size(TPack *pack);

#ifdef __cplusplus
}
#endif
