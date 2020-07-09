#include "lib.h"
#include "impl.h"

#include <vector>
#include <unordered_map>
#include <array>
#include <iostream>

extern "C" void print_one(TObject *val);

namespace lua {
namespace {

struct LuaHash {
  std::size_t operator()(TObject *val) const {
    switch (lua_get_type(val)) {
    case NIL:
      return std::hash<std::nullptr_t>{}(nullptr);
    case BOOL:
      return std::hash<bool>{}(lua_get_bool_val(val));
    case NUM:
      if (lua_is_int(val)) {
        return std::hash<int64_t>{}(lua_get_int64_val(val));
      } else {
        return std::hash<double>{}(lua_get_double_val(val));
      }
    case STR:
      return std::hash<std::string>{}(as_std_string(val));
    default:
      return std::hash<uint64_t>{}(val->u);
    }
  }
};

struct LuaEq {
  static bool compare(TObject *lhs, TObject *rhs) {
    switch (lua_get_type(lhs)) {
    case NIL:
      return true;
    case BOOL:
      return lua_get_bool_val(lhs) == lua_get_bool_val(rhs);
    case NUM:
      if (lua_is_int(lhs) != lua_is_int(rhs)) {
        return false;
      } else if (lua_is_int(lhs)) {
        return lua_get_int64_val(lhs) == lua_get_int64_val(rhs);
      } else {
        return lua_get_double_val(lhs) == lua_get_double_val(rhs);
      }
    case STR:
      return as_std_string(lhs) == as_std_string(rhs);
    default:
      return lhs->u == rhs->u;
    }
  }

  bool operator()(TObject *lhs, TObject *rhs) const {
    if (lua_get_type(lhs) != lua_get_type(rhs)) {
      return false;
    }
    return compare(lhs, rhs);
  }
};

struct LuaTable {
  static constexpr std::size_t PREALLOC = 4;
  std::array<TObject, PREALLOC> prealloc{};
  std::vector<TObject> trailing{};
  std::unordered_map<TObject *, TObject *, LuaHash, LuaEq> table;

  auto *prealloc_get_or_alloc(int64_t iv) {
    return &prealloc[iv];
  }

  auto *list_get_or_alloc(int64_t iv) {
    --iv;
    if (iv < PREALLOC) {
      return prealloc_get_or_alloc(iv);
    }
    iv -= PREALLOC;
    if (trailing.size() <= iv) {
      trailing.resize(iv * 2);
    }
    return &trailing[iv];
  }

  auto *get_or_alloc(TObject *key) {
    if (lua_get_type(key) == NUM && lua_is_int(key)) {
      if (auto iv = lua_get_int64_val(key); iv > 0) {
        return list_get_or_alloc(iv);
      }
    }
    if (auto it = table.find(key); it != table.end()) {
      return it->second;
    } else {
      auto *nil = lua_alloc();
      lua_set_type(nil, NIL);
      table.emplace_hint(it, key, nil);
      return nil;
    }
  }

  void prealloc_insert_or_assign(int64_t iv, TObject *val) {
    prealloc[iv] = *val;
  }

  void list_insert_or_assign(int64_t iv, TObject *val) {
    --iv;
    if (iv < PREALLOC) {
      prealloc_insert_or_assign(iv, val);
      return;
    }
    iv -= PREALLOC;
    if (trailing.size() <= iv) {
      trailing.resize(iv * 2);
    }
    trailing[iv] = *val;
  }

  void insert_or_assign(TObject *key, TObject *val) {
    if (lua_get_type(key) == NUM && lua_is_int(key)) {
      if (auto iv = lua_get_int64_val(key); iv > 0) {
        return list_insert_or_assign(iv, val);
      }
    }
    auto *nval = lua_alloc();
    *nval = *val;
    table.insert_or_assign(key, nval);
  }

  int64_t get_list_size() {
    int64_t size = 0;
    for (size_t i = 0; i < PREALLOC; ++i) {
      if (prealloc[i].type != NIL) {
        ++size;
      } else {
        return size;
      }
    }
    for (size_t i = 0; i < trailing.size(); ++i) {
      if (trailing[i].type != NIL) {
        ++size;
      } else {
        return size;
      }
    }
    return size;
  }
};

} // end anonymous namespace

std::string &as_std_string(TObject *val) {
  return *((std::string *) val->gc->pstring);
}

} // end namespace lua

extern "C" {

static TObject s_pool[16*1024*1024];
static TObject *ptr = s_pool;

TObject *lua_alloc(void) {
  return ptr++;
}

TPack g_ret_pack_impl;
TPack g_arg_pack_impl;
TPack *g_ret_pack = &g_ret_pack_impl;
TPack *g_arg_pack = &g_arg_pack_impl;

void lua_init_table_impl(TObject *tbl) {
  tbl->gc->ptable = new lua::LuaTable;
}

void lua_table_set_impl(TObject *tbl, TObject *key, TObject *val) {
  ((lua::LuaTable *) tbl->gc->ptable)->insert_or_assign(key, val);
}

void lua_table_set_prealloc_impl(TObject *tbl, int64_t iv, TObject *val) {
  ((lua::LuaTable *) tbl->gc->ptable)->prealloc_insert_or_assign(iv, val);
}

TObject *lua_table_get_impl(TObject *tbl, TObject *key) {
  return ((lua::LuaTable *) tbl->gc->ptable)->get_or_alloc(key);
}

TObject *lua_table_get_prealloc_impl(TObject *tbl, int64_t iv) {
  return ((lua::LuaTable *) tbl->gc->ptable)->prealloc_get_or_alloc(iv);
}

TObject *lua_load_string_impl(const char *data, uint64_t len) {
  auto *ret = lua_alloc();
  lua_set_type(ret, STR);
  lua_alloc_gc(ret);
  ret->gc->pstring = new std::string{data, len};
  return ret;
}

bool lua_eq_impl(TObject *lhs, TObject *rhs) {
  // already verified as same type
  return lua::LuaEq::compare(lhs, rhs);
}

int64_t lua_list_size_impl(TObject *tbl) {
  return ((lua::LuaTable *) tbl->gc->ptable)->get_list_size();
}

void lua_strcat_impl(TObject *dest, TObject *lhs, TObject *rhs) {
  dest->gc->pstring = new std::string{lua::as_std_string(lhs) +
                                      lua::as_std_string(rhs)};
}

int64_t ipow_impl(int64_t base, int64_t exp) {
  int64_t result = 1;
  for (;;) {
    if (exp & 1) {
      result *= base;
    }
    exp >>= 1;
    if (!exp) {
      break;
    }
    base *= base;
  }
  return result;
}

}
