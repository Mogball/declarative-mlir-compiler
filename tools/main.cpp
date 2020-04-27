#include "dmc/Dynamic/DynamicContext.h"

#include <iostream>

using namespace mlir;
using namespace dmc;

int main() {
  MLIRContext mlirContext;
  DynamicContext ctx{&mlirContext};

  auto *dialectTest0 = ctx.createDynamicDialect("test0");
  auto *dialectTest1 = ctx.createDynamicDialect("test1");
  assert(dialectTest0 == mlirContext.getRegisteredDialect("test0"));
  assert(dialectTest1 == mlirContext.getRegisteredDialect("test1"));
  assert(nullptr == mlirContext.getRegisteredDialect("test2"));
  std::cout << "All good!" << std::endl;
}
