#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int64_t LuaInteger;
typedef double LuaReal;

enum LuaType {
  Nil, Bool, Number, String, Table, Function, UserData, Thread
};

struct GCObject {
  void *todo;
};

enum Kind {
  Integer, Real
};

struct LuaNumber {
  union {
    LuaInteger iv;
    LuaReal fp;
  };
  enum Kind kind;
};

union Value {
  struct GCObject *gc;
  void *userdata;
  struct LuaNumber number;
  bool boolean;
};

typedef struct TObject {
  enum LuaType type;
  union Value value;
} TObject;

extern TObject lua_add(TObject *lhs, TObject *rhs);
extern TObject lua_sub(TObject *lhs, TObject *rhs);
extern TObject lua_eq(TObject *lhs, TObject *rhs);
extern TObject lua_neq(TObject *lhs, TObject *rhs);

extern TObject lua_get_nil(void);
extern TObject lua_new_table(void);
// string is not null-terminated
extern TObject lua_get_string(const char *data, int32_t len);

extern TObject lua_wrap_int(LuaInteger val);
extern TObject lua_wrap_real(LuaReal val);
extern TObject lua_wrap_bool(bool boolean);
extern LuaInteger lua_unwrap_int(TObject *val);
extern LuaReal lua_unwrap_real(TObject *val);
extern bool lua_unwrap_bool(TObject *val);

extern TObject lua_typeof(TObject *val);

extern TObject lua_table_get(TObject *tbl, TObject *key);
extern void lua_table_set(TObject *tbl, TObject *key, TObject *val);
extern TObject lua_table_size(TObject *tbl);

#ifdef __cplusplus
}
#endif
