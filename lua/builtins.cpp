#include "impl.h"

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
        std::cout << std::hex << ((unsigned char *) &val->gc->fcn_addr)[i];
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

TObject *construct_builtin_print() {
  TObject *ret = lua_alloc();
  lua_set_type(ret, FCN);
  lua_alloc_gc(ret);
  lua_set_fcn_addr(ret, &fcn_builtin_print);
  lua_set_capture_pack(ret, nullptr);
  return ret;
}

TObject *construct_builtin_string() {
  TObject *ret = lua_alloc();
  lua_set_type(ret, NIL);
  return ret;
}

} // end anonymous namespace
} // end namespace lua

extern "C" {

TObject *builtin_print = lua::construct_builtin_print();
TObject *builtin_string = lua::construct_builtin_string();

}
