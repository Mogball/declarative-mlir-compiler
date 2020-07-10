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
  std::size_t operator()(TObject val) const {
    switch (val.type) {
    case NIL:
      return std::hash<std::nullptr_t>{}(nullptr);
    case BOOL:
      return std::hash<bool>{}(val.b);
    case NUM:
      return std::hash<double>{}(val.num);
    case STR:
      return std::hash<std::string>{}(as_std_string(val));
    default:
      return std::hash<uint64_t>{}(val->u);
    }
  }
};

struct LuaEq {
  static bool compare(TObject lhs, TObject rhs) {
    switch (lhs.type) {
    case NIL:
      return true;
    case BOOL:
      return lhs.b == rhs.b;
    case NUM:
      return lhs.num == rhs.num;
    case STR:
      return as_std_string(lhs) == as_std_string(rhs);
    default:
      return lhs.u = rhs.u;
    }
  }

  bool operator()(TObject lhs, TObject rhs) const {
    return lhs.type == rhs.type && compare(lhs, rhs);
  }
};

struct LuaTable {
  static constexpr std::size_t PREALLOC = 4;
  std::array<TObject, PREALLOC> prealloc{};
  std::vector<TObject> trailing{};
  std::unordered_map<TObject, TObject, LuaHash, LuaEq> table;

  TObject prealloc_get_or_alloc(int64_t iv) {
    return prealloc[iv];
  }

  TObject list_get_or_alloc(int64_t iv) {
    --iv;
    if (iv < PREALLOC) {
      return prealloc_get_or_alloc(iv);
    }
    iv -= PREALLOC;
    if (trailing.size() <= iv) {
      trailing.resize(iv * 2);
    }
    return trailing[iv];
  }

  TObject get_or_alloc(TObject key) {
    if (key.type == NUM) {
      auto num = key.num;
      auto iv = (int64_t) num;
      if (iv == num && iv > 0) {
        return list_get_or_alloc(iv);
      }
    }
    if (auto it = table.find(key); it != table.end()) {
      return it->second;
    } else {
      TObject nil{NIL};
      table.emplace_hint(it, key, nil);
      return nil;
    }
  }

  void prealloc_insert_or_assign(int64_t iv, TObject val) {
    prealloc[iv] = val;
  }

  void list_insert_or_assign(int64_t iv, TObject val) {
    --iv;
    if (iv < PREALLOC) {
      prealloc_insert_or_assign(iv, val);
      return;
    }
    iv -= PREALLOC;
    if (trailing.size() <= iv) {
      trailing.resize(iv * 2);
    }
    trailing[iv] = val;
  }

  void insert_or_assign(TObject key, TObject val) {
    if (key.type == NUM) {
      auto num = key.num;
      auto iv = (int64_t) num;
      if (iv == num && iv > 0) {
        list_insert_or_assign(iv, val);
        return;
      }
    }
    table.insert_or_assign(key, val);
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

std::string &as_std_string(TObject val) {
  return *((std::string *) val.gc->pstring);
}

} // end namespace lua

extern "C" {

static TPack g_ret_pack_impl;
static TPack g_arg_pack_impl;
TPack *g_ret_pack = &g_ret_pack_impl;
TPack *g_arg_pack = &g_arg_pack_impl;

void lua_init_table_impl(TObject tbl) {
  tbl.gc->ptable = new lua::LuaTable;
}
void lua_table_set_impl(TObject tbl, TObject key, TObject val) {
  ((lua::LuaTable *) tbl.gc->ptable)->insert_or_assign(key, val);
}
TObject lua_table_get_impl(TObject tbl, TObject key) {
  return ((lua::LuaTable *) tbl.gc->ptable)->get_or_alloc(key);
}

void lua_table_set_prealloc_impl(TObject tbl, int64_t iv, TObject val) {
  ((lua::LuaTable *) tbl.gc->ptable)->prealloc_insert_or_assign(iv, val);
}
TObject *lua_table_get_prealloc_impl(TObject tbl, int64_t iv) {
  return ((lua::LuaTable *) tbl.gc->ptable)->prealloc_get_or_alloc(iv);
}

TObject lua_load_string_impl(const char *data, uint64_t len) {
  TObject ret{STR};
  ret.gc = malloc(sizeof(TComplex));
  ret.gc->pstring = new std::string{data, len};
  return ret;
}

bool lua_eq_impl(TObject lhs, TObject rhs) {
  // already verified as same type
  return lua::LuaEq::compare(lhs, rhs);
}

int64_t lua_list_size_impl(TObject tbl) {
  return ((lua::LuaTable *) tbl.gc->ptable)->get_list_size();
}

void lua_strcat_impl(TObject *dest, TObject lhs, TObject rhs) {
  dest->gc->pstring = new std::string{lua::as_std_string(lhs) +
                                      lua::as_std_string(rhs)};
}

}
