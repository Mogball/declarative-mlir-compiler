#include "lib.h"

extern TPack *lua_main(void);

int main() {
  TPack *pack = lua_main();
  TObject *code = lua_pack_pull_one(pack);
  switch (lua_get_type(code)) {
  case NIL:
    return 0;
  case BOOL:
    return (int) lua_get_bool_val(code);
  case NUM:
    return (int) lua_get_double_val(code);
  default:
    return 0;
  }
}
