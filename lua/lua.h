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

struct GCObject;

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

extern void lua_add(TObject *sret, TObject *lhs, TObject *rhs);
extern void lua_sub(TObject *sret, TObject *lhs, TObject *rhs);
extern void lua_eq(TObject *sret, TObject *lhs, TObject *rhs);
extern void lua_neq(TObject *sret, TObject *lhs, TObject *rhs);

extern void lua_get_nil(TObject *sret);
extern void lua_new_table(TObject *sret);
// string is not null-terminated
extern void lua_get_string(TObject *sret, const char *data, int32_t len);

extern void lua_wrap_int(TObject *sret, LuaInteger val);
extern void lua_wrap_real(TObject *sret, LuaReal val);
extern void lua_wrap_bool(TObject *sret, bool boolean);
extern LuaInteger lua_unwrap_int(TObject *val);
extern LuaReal lua_unwrap_real(TObject *val);
extern bool lua_unwrap_bool(TObject *val);

extern void lua_typeof(TObject *sret, TObject *val);

extern void lua_table_get(TObject *sret, TObject *tbl, TObject *key);
extern void lua_table_set(TObject *tbl, TObject *key, TObject *val);
extern void lua_table_size(TObject *sret, TObject *tbl);

#ifdef __cplusplus
}
#endif
