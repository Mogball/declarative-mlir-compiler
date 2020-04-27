#include "dmc/Dynamic/DynamicDialect.h"
#include "dmc/Dynamic/TypeIDAllocator.h"

#include <iostream>

using namespace mlir;
using namespace dmc;

int main() {
  auto *typeIdAlloc = getFixedTypeIDAllocator();
  Dialect::registerDialectAllocator(typeIdAlloc->allocateID(), [](MLIRContext *ctx) {
    new DynamicDialect{"test", ctx};
  });
  Dialect::registerDialectAllocator(typeIdAlloc->allocateID(), [](MLIRContext *ctx) {
    new DynamicDialect{"test1", ctx};
  });
}
