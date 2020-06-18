#include "lua.h"

#include <iostream>
#include <cstddef>

extern "C" {

void lua_add(TObject *sret, TObject *lhs, TObject *rhs) {
}

void lua_sub(TObject *sret, TObject *lhs, TObject *rhs) {
}

void lua_eq(TObject *sret, TObject *lhs, TObject *rhs) {
}

void lua_neq(TObject *sret, TObject *lhs, TObject *rhs) {
}


void lua_get_nil(TObject *sret) {
  sret->type = LuaType::Nil;
}

void lua_new_table(TObject *sret) {
}

void lua_get_string(TObject *sret, const char *data, int32_t len) {
}


void lua_wrap_int(TObject *sret, LuaInteger val) {
  sret->type = LuaType::Number;
  sret->value.number.kind = Kind::Integer;
  sret->value.number.iv = val;
}

void lua_wrap_real(TObject *sret, LuaReal val) {
  sret->type = LuaType::Number;
  sret->value.number.kind = Kind::Real;
  sret->value.number.fp = val;
}

void lua_wrap_bool(TObject *sret, bool boolean) {
  sret->type = LuaType::Bool;
  sret->value.boolean = boolean;
}

LuaInteger lua_unwrap_int(TObject *val) {
  if (val->type != LuaType::Number) {
    throw std::invalid_argument{"Lua object is not a number"};
  }
  if (val->value.number.kind != Kind::Integer) {
    throw std::invalid_argument{"Lua number is not an integer"};
  }
  return val->value.number.iv;
}

LuaReal lua_unwrap_real(TObject *val) {
  if (val->type != LuaType::Number) {
    throw std::invalid_argument{"Lua object is not a number"};
  }
  if (val->value.number.kind != Kind::Real) {
    throw std::invalid_argument{"Lua number is not a real"};
  }
  return val->value.number.fp;
}

bool lua_unwrap_bool(TObject *val) {
  if (val->type != LuaType::Bool) {
    throw std::invalid_argument{"Lua object is not a boolean"};
  }
  return val->value.boolean;
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
  using namespace std;

  switch (val->type) {
  case LuaType::Nil:
    cout << "nil" << endl;
    break;
  case LuaType::Bool:
    cout << (val->value.boolean ? "true" : "false") << endl;
    break;
  case LuaType::Number:
    auto &number = val->value.number;
    switch (number.kind) {
    case Kind::Integer:
      cout << number.iv << endl;
      break;
    default:
      cout << number.fp << endl;
      break;
    }
    break;
  default:
    cout << "unknown" << endl;
    break;
  }
}

void random_string_or_int(TObject *sret, size_t len) {
  lua_wrap_int(sret, 4);
}

}
