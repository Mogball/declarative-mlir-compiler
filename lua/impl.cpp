#include "lib.h"
#include "impl.h"

#include <vector>
#include <unordered_map>
#include <array>
#include <iostream>
#include <cassert>

static_assert(sizeof(TObject) == 16, "expected TObject to be 16 bytes");

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
      return std::hash<std::string>{}(*((std::string *) val.impl));
    default:
      return std::hash<int64_t>{}(val.u);
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
      return *((std::string *) lhs.impl) == *((std::string *) rhs.impl);
    default:
      return lhs.u == rhs.u;
    }
  }

  bool operator()(TObject lhs, TObject rhs) const {
    return lhs.type == rhs.type && compare(lhs, rhs);
  }
};

static constexpr std::size_t PREALLOC = 16;
using prealloc_t = std::array<TObject, PREALLOC>;
static_assert(sizeof(prealloc_t) == PREALLOC * sizeof(TObject),
              "mismatched prealloc size");

struct LuaTable {
  prealloc_t prealloc;
  //std::vector<TObject> trailing;
  //std::unordered_map<TObject, TObject, LuaHash, LuaEq> table;

  TObject prealloc_get_or_alloc(int64_t iv) {
    return prealloc[iv];
  }

  //TObject list_get_or_alloc(int64_t iv) {
  //  --iv;
  //  if (iv < PREALLOC) {
  //    return prealloc_get_or_alloc(iv);
  //  }
  //  iv -= PREALLOC;
  //  if (trailing.size() <= iv) {
  //    trailing.resize(1 + iv * 2);
  //  }
  //  return trailing[iv];
  //}

  TObject get_or_alloc(TObject key) {
    return prealloc_get_or_alloc(key.u - 1);
    //if (key.type == INT) {
    //  auto iv = key.u;
    //  if (iv > 0) {
    //    return list_get_or_alloc(iv);
    //  }
    //}
    //if (auto it = table.find(key); it != table.end()) {
    //  return it->second;
    //} else {
    //  TObject nil{NIL};
    //  table.emplace_hint(it, key, nil);
    //  return nil;
    //}
  }

  void prealloc_insert_or_assign(int64_t iv, TObject val) {
    prealloc[iv] = val;
  }

  //void list_insert_or_assign(int64_t iv, TObject val) {
  //  --iv;
  //  if (iv < PREALLOC) {
  //    prealloc_insert_or_assign(iv, val);
  //    return;
  //  }
  //  iv -= PREALLOC;
  //  if (trailing.size() <= iv) {
  //    trailing.resize(1 + iv * 2);
  //  }
  //  trailing[iv] = val;
  //}

  void insert_or_assign(TObject key, TObject val) {
    prealloc_insert_or_assign(key.u - 1, val);
    //if (key.type == INT) {
    //  auto iv = key.u;
    //  if (iv > 0) {
    //    list_insert_or_assign(iv, val);
    //    return;
    //  }
    //}
    //table.insert_or_assign(key, val);
  }

  //int64_t get_list_size() {
  //  int64_t size = 0;
  //  for (size_t i = 0; i < PREALLOC; ++i) {
  //    if (prealloc[i].type != NIL) {
  //      ++size;
  //    } else {
  //      return size;
  //    }
  //  }
  //  for (size_t i = 0; i < trailing.size(); ++i) {
  //    if (trailing[i].type != NIL) {
  //      ++size;
  //    } else {
  //      return size;
  //    }
  //  }
  //  return size;
  //}
};

} // end anonymous namespace
} // end namespace lua

extern "C" {

void *lua_make_fcn_impl(lua_fcn_t addr, TCapture capture) {
  return new TClosure{addr, capture};
}

void *lua_new_table_impl(void) {
  return (void *) new lua::LuaTable;
}
void lua_table_set_impl(void *impl, TObject key, TObject val) {
  ((lua::LuaTable *) impl)->insert_or_assign(key, val);
}
TObject lua_table_get_impl(void *impl, TObject key) {
  return ((lua::LuaTable *) impl)->get_or_alloc(key);
}
void lua_table_set_prealloc_impl(void *impl, int64_t iv, TObject val) {
  ((lua::LuaTable *) impl)->prealloc_insert_or_assign(iv, val);
}
TObject lua_table_get_prealloc_impl(void *impl, int64_t iv) {
  return ((lua::LuaTable *) impl)->prealloc_get_or_alloc(iv);
}

int64_t lua_list_size_impl(void *impl) {
  assert(false);
  //return ((lua::LuaTable *) impl)->get_list_size();
}

void *lua_load_string_impl(const char *data, uint64_t len) {
  return new std::string{data, len};
}

bool lua_eq_impl(TObject lhs, TObject rhs) {
  // already verified as same type
  return lua::LuaEq::compare(lhs, rhs);
}

TObject lua_strcat_impl(void* lhs, void *rhs) {
  TObject ret;
  ret.type = STR;
  auto catted = *((std::string *) lhs) + *((std::string *) rhs);
  ret.impl = new std::string{std::move(catted)};
  return ret;
}

int64_t ipow(int64_t x, int64_t p)
{
  if (p == 0) return 1;
  if (p == 1) return x;

  int tmp = ipow(x, p/2);
  if (p%2 == 0) return tmp * tmp;
  else return x * tmp * tmp;
}

void print_i32(int32_t type) {
  std::cout << type << std::endl;
}

}
