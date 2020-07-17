#include "lib.h"
#include <stdlib.h>
#include <stdio.h>

int main() {
  int a = 10;
  for (int i = 10; i < 100; ++i) {
    a = a * 2 + a;
    int g = a;
    int q;
    if (i == 55 && g > 22) {
      q = 44;
    } else {
      q = a + 44;
    }
    a += q;
  }
  printf("%d\n", a);
}
