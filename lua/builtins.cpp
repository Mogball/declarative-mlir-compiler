#include "lib.h"

#include <cstdio>
#include <cstdlib>
#include <cinttypes>

extern "C" {

TPack *fcn_builtin_print(TPack *, TPack *pack) {
  while (pack->idx != pack->size) {
    TObject *val = lua_pack_pull_one(pack);
    switch (lua_get_type(val)) {
    case NIL:
      printf("%-8s", "nil");
      break;
    case BOOL:
      if (val->b) {
        printf("%-8s", "true");
      } else {
        printf("%-8s", "false");
      }
      break;
    case NUM:
      if (lua_is_int(val)) {
        printf("%-8" PRId64, lua_get_int64_val(val));
      } else {
        printf("%-8f", lua_get_double_val(val));
      }
      break;
    case STR:
      printf("string");
      break;
    case TBL:
      printf("table");
      break;
    case FCN:
      printf("function: 0x");
      for (size_t i = 0; i < sizeof(lua_fcn_t); ++i) {
        printf("%.2x", ((unsigned char *) &val->gc->fcn_addr)[i]);
      }
      break;
    }
  }
  printf("\n");

  TPack *ret = lua_new_pack(1);
  TObject *nil = lua_alloc();
  lua_set_type(nil, NIL);
  lua_pack_push(ret, nil);
  return ret;
}

TObject *construct_builtin_print() {
  TObject *ret = lua_alloc();
  lua_set_type(ret, FCN);
  ret->gc = (TComplex *) malloc(sizeof(TComplex));
  ret->gc->fcn_addr = &fcn_builtin_print;
  ret->gc->cap_pack = nullptr;
  return ret;
}

TObject *builtin_print = construct_builtin_print();

}
