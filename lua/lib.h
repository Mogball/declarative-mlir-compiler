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

typedef TPack *(*lua_fcn_t)(TPack *, TPack *);
typedef void *lua_opaque_table_t;
typedef void *lua_opaque_string_t;

typedef union Complex {
  struct {
    lua_fcn_t fcn_addr;
    TPack *cap_pack;
  };
  lua_opaque_table_t ptable;
  lua_opaque_string_t pstring;
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

TObject *lua_alloc(void);
void lua_alloc_gc(TObject *val);
int16_t lua_get_type(TObject *val);
void lua_set_type(TObject *val, int16_t ty);
bool lua_get_bool_val(TObject *val);
void lua_set_bool_val(TObject *val, bool b);
int64_t lua_get_int64_val(TObject *val);
void lua_set_int64_val(TObject *val, int64_t iv);
double lua_get_double_val(TObject *val);
void lua_set_double_val(TObject *val, double fp);
lua_fcn_t lua_get_fcn_addr(TObject *val);
void lua_set_fcn_addr(TObject *val, lua_fcn_t fcn_addr);
TPack *lua_get_capture_pack(TObject *val);
void lua_set_capture_pack(TObject *val, TPack *pack);
uint64_t lua_get_value_union(TObject *val);
void lua_set_value_union(TObject *val, uint64_t u);
bool lua_is_int(TObject *val);
TPack *lua_new_pack(int64_t size);
void lua_delete_pack(TPack *pack);
void lua_pack_push(TPack *pack, TObject *val);
TObject *lua_pack_pull_one(TPack *pack);
void lua_pack_push_all(TPack *pack, TPack *vals);
int64_t lua_pack_get_size(TPack *pack);
void lua_pack_rewind(TPack *pack);
void lua_init_table(TObject *tbl);
void lua_table_set(TObject *tbl, TObject *key, TObject *val);
TObject *lua_table_get(TObject *tbl, TObject *key);
void lua_table_set(TObject *tbl, TObject *key, TObject *val);
TObject *lua_load_string(const char *data, uint64_t len);

TObject *lua_list_size(TObject *tbl);
TObject *lua_add(TObject *lhs, TObject *rhs);

#ifdef __cplusplus
}
#endif
