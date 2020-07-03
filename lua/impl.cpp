#include "lib.h"
#include "impl.h"

#include <vector>
#include <unordered_map>
#include <array>

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
  bool operator()(TObject *lhs, TObject *rhs) const {
    if (lua_get_type(lhs) != lua_get_type(rhs)) {
      return false;
    }
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
};

struct LuaTable {
  static constexpr std::size_t PREALLOC = 256;
  std::array<TObject *, PREALLOC> prealloc{};
  std::vector<TObject *> trailing{};
  std::unordered_map<TObject *, TObject *, LuaHash, LuaEq> table;

  auto *list_get_or_alloc(int64_t iv) {
    --iv;
    if (iv < PREALLOC) {
      auto *ret = prealloc[iv];
      if (ret) {
        return ret;
      }
      auto *nil = lua_alloc();
      lua_set_type(nil, NIL);
      prealloc[iv] = nil;
      return nil;
    }
    iv -= PREALLOC;
    if (trailing.size() <= iv) {
      trailing.resize(iv * 2);
    }
    auto ret = trailing[iv];
    if (ret) {
      return ret;
    }
    auto *nil = lua_alloc();
    lua_set_type(nil, NIL);
    trailing[iv] = nil;
    return nil;
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

  void list_insert_or_assign(int64_t iv, TObject *val) {
    --iv;
    if (iv < PREALLOC) {
      prealloc[iv] = val;
      return;
    }
    iv -= PREALLOC;
    if (trailing.size() <= iv) {
      trailing.resize(iv * 2);
    }
    trailing[iv] = val;
  }

  void insert_or_assign(TObject *key, TObject *val) {
    if (lua_get_type(key) == NUM && lua_is_int(key)) {
      if (auto iv = lua_get_int64_val(key); iv > 0) {
        return list_insert_or_assign(iv, val);
      }
    }
    table.insert_or_assign(key, val);
  }
};

} // end anonymous namespace

std::string &as_std_string(TObject *val) {
  return *((std::string *) val->gc->pstring);
}

} // end namespace lua

extern "C" {

void lua_init_table_impl(TObject *tbl) {
  tbl->gc->ptable = new lua::LuaTable;
}

void lua_table_set_impl(TObject *tbl, TObject *key, TObject *val) {
  ((lua::LuaTable *) tbl->gc->ptable)->insert_or_assign(key, val);
}

TObject *lua_table_get_impl(TObject *tbl, TObject *key) {
  return ((lua::LuaTable *) tbl->gc->ptable)->get_or_alloc(key);
}

TObject *lua_load_string_impl(const char *data, uint64_t len) {
  auto *ret = lua_alloc();
  lua_set_type(ret, STR);
  lua_alloc_gc(ret);
  ret->gc->pstring = new std::string{data, len};
  return ret;
}

}
