#include <stdint.h>
#include <stdbool.h>

enum LuaType {
  Nil, Bool, Number, String, Table, Function, UserData, Thread
};

struct GCObject {};

enum Kind {
  Integer, Floating
};

struct LuaNumber {
  union {
    int64_t iv;
    double fp;
  };
  enum Kind kind;
};

union Value {
  struct GCObject *gc;
  void *userdata;
  struct LuaNumber number;
  bool boolean;
};

struct TObject {
  enum LuaType type;
  union Value value;
};

extern struct TObject lua_add(struct TObject *lhs, struct TObject *rhs);
extern struct TObject lua_sub(struct TObject *lhs, struct TObject *rhs);

extern struct TObject lua_wrap_int(int64_t val);
extern struct TObject lua_wrap_float(double val);

void hold() {
  (void) lua_add(0, 0);
  (void) lua_sub(0, 0);

  (void) lua_wrap_int(0);
  (void) lua_wrap_float(0);
}
