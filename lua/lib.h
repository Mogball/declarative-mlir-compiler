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
  NIL,
  BOOL,
  NUM,
  STR,
  TBL,
  FCN
};

struct Object;

typedef struct Pack {
  int32_t size;
  struct Object *objs;
} TPack;

typedef struct Object **TCapture;

typedef TPack (*lua_fcn_t)(TCapture, TPack);

typedef struct Closure {
  lua_fcn_t addr;
  TCapture capture;
} TClosure;

typedef struct Object {
  int32_t type;
  union {
    uint64_t u;

    bool b;
    double num;
    void *impl;
  };
} TObject;


/*******************************************************************************
 * Simple Value Manipulation
 ******************************************************************************/

TObject lua_nil(void);

TObject lua_alloc(void);
void lua_copy(TObject *ptr, TObject val);

int32_t lua_get_type(TObject val);
void lua_set_type(TObject *ptr, int32_t ty);

bool lua_get_bool_val(TObject val);
void lua_set_bool_val(TObject *ptr, bool b);

double lua_get_double_val(TObject val);
void lua_set_double_val(TObject *ptr, double fp);

lua_fcn_t lua_get_fcn_addr(TObject val);
void lua_set_fcn_addr(TObject val, lua_fcn_t fcn_addr);

TCapture lua_get_capture_pack(TObject val);
void lua_set_capture_pack(TObject val, TCapture capture);

uint64_t lua_get_value_union(TObject val);
void lua_set_value_union(TObject *ptr, uint64_t u);

/*******************************************************************************
 * Pack Manipulation
 ******************************************************************************/

TCapture lua_new_capture(int32_t size);
void lua_add_capture(TCapture capture, TObject *ptr, int32_t idx);

TPack lua_get_ret_pack(int32_t size);
TPack lua_get_arg_pack(int32_t size);

void lua_pack_insert(TPack pack, TObject val, int32_t idx);
void lua_pack_insert_all(TPack pack, TPack tail, int32_t idx);
TObject lua_pack_get(TPack pack, int32_t idx);
int32_t lua_pack_get_size(TPack pack);

/*******************************************************************************
 * Tables and Strings
 ******************************************************************************/

TObject lua_new_table(void);
void lua_table_set(TObject tbl, TObject key, TObject val);
TObject lua_table_get(TObject tbl, TObject key);
void lua_table_set(TObject tbl, TObject key, TObject val);

TObject lua_list_size(TObject tbl);
TObject lua_load_string(const char *data, uint64_t len);

#ifdef __cplusplus
}
#endif
