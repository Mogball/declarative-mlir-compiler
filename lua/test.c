#include "lib.h"
#include <stdlib.h>

uint64_t g_ptr;

TPack get_arg_pack(int32_t sz) {
  g_ptr = realloc(g_ptr, sz);

  TPack ret;
  ret.size = sz;
  ret.objs = g_ptr;
  return ret;
}
void release(void) {
  free((void *)g_ptr);
}
