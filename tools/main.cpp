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

  dialectTest0->createDynamicOp("opA");
  dialectTest0->createDynamicOp("opB");
  dialectTest1->createDynamicOp("opC");
  dialectTest1->createDynamicOp("opD");
  std::cout << "Ops registered" << std::endl;

  auto *opA = AbstractOperation::lookup("test0.opA", &mlirContext);
  auto *opB = AbstractOperation::lookup("test0.opB", &mlirContext);
  auto *opC = AbstractOperation::lookup("test1.opC", &mlirContext);
  auto *opD = AbstractOperation::lookup("test1.opD", &mlirContext);

  assert(opA != nullptr);
  assert(opB != nullptr);
  assert(opC != nullptr);
  assert(opD != nullptr);
  assert(AbstractOperation::lookup("test0.opC", &mlirContext) == nullptr);
  std::cout << "All ops found" << std::endl;
}
