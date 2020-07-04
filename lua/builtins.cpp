#include "impl.h"
#include "rx-cpp/src/lua-str.h"

#include <iostream>
#include <iomanip>

namespace lua {
namespace {

void formatOutput(std::ostream &os) {
  os << std::left << std::setw(8);
}

TPack *fcn_builtin_print(TPack *, TPack *pack) {
  while (pack->idx != pack->size) {
    auto *val = lua_pack_pull_one(pack);
    switch (lua_get_type(val)) {
    case NIL:
      formatOutput(std::cout);
      std::cout << "nil";
      break;
    case BOOL:
      formatOutput(std::cout);
      if (lua_get_bool_val(val)) {
        std::cout << "true";
      } else {
        std::cout << "false";
      }
      break;
    case NUM:
      formatOutput(std::cout);
      if (lua_is_int(val)) {
        std::cout << lua_get_int64_val(val);
      } else {
        std::cout << lua_get_double_val(val);
      }
      break;
    case STR:
      formatOutput(std::cout);
      std::cout << lua::as_std_string(val);
      break;
    case TBL:
      std::cout << "table: 0x";
      formatOutput(std::cout);
      std::cout << std::hex << val->gc->ptable;
      break;
    case FCN:
      std::cout << "function: 0x";
      for (unsigned i = 0; i < sizeof(lua_fcn_t); ++i) {
        if (i == sizeof(lua_fcn_t) - 1) {
          formatOutput(std::cout);
        }
        std::cout << std::hex << (int)((unsigned char *) &val->gc->fcn_addr)[i];
      }
      break;
    }
  }
  std::cout << std::endl;

  auto *ret = lua_new_pack(1);
  auto *nil = lua_alloc();
  lua_set_type(nil, NIL);
  lua_pack_push(ret, nil);
  return ret;
}

TPack *fcn_builtin_string_find(TPack *, TPack *pack) {
  auto *text = lua_pack_pull_one(pack);
  auto *pattern = lua_pack_pull_one(pack);
  auto *pos = lua_pack_pull_one(pack);
  auto &textStr = as_std_string(text);
  auto &patStr = as_std_string(pattern);
  int64_t offset = 0;
  if (lua_get_type(pos) != NIL) {
    offset = lua_get_int64_val(pos);
    if (offset > 0) {
      --offset;
      if (offset >= textStr.size()) {
        offset = textStr.size();
      }
    } else if (offset < 0) {
      offset = (int64_t) textStr.size() + offset;
      if (offset < 0) {
        offset = 0;
      }
    }
  }
  LuaMatch m;
  auto n = str_match(textStr.c_str() + offset, textStr.size() - offset,
                     patStr.c_str(), &m);
  if (n == 0) {
    auto *ret = lua_new_pack(1);
    auto *nil = lua_alloc();
    lua_set_type(nil, NIL);
    lua_pack_push(ret, nil);
    return ret;
  }
  auto *ret = lua_new_pack(2);
  auto *start = lua_alloc();
  lua_set_type(start, NUM);
  lua_set_int64_val(start, m.start + 1 + offset);
  lua_pack_push(ret, start);
  auto *end = lua_alloc();
  lua_set_type(end, NUM);
  lua_set_int64_val(end, m.end + offset);
  lua_pack_push(ret, end);
  return ret;
}

TObject *construct_builtin_print(void) {
  TObject *print = lua_alloc();
  lua_set_type(print, FCN);
  lua_alloc_gc(print);
  lua_set_fcn_addr(print, &fcn_builtin_print);
  lua_set_capture_pack(print, nullptr);
  return print;
}

TObject *construct_builtin_string(void) {
  TObject *string = lua_alloc();
  lua_set_type(string, TBL);
  lua_alloc_gc(string);
  lua_init_table(string);
  {
    TObject *find = lua_alloc();
    lua_set_type(find, FCN);
    lua_alloc_gc(find);
    lua_set_fcn_addr(find, &fcn_builtin_string_find);
    lua_set_capture_pack(find, nullptr);
    auto *key = lua_load_string("find", 4);
    lua_table_set(string, key, find);
  }
  return string;
}

} // end anonymous namespace
} // end namespace lua

extern "C" {

TObject *builtin_print = lua::construct_builtin_print();
TObject *builtin_string = lua::construct_builtin_string();

}
