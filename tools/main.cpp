#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/IO/ModuleWriter.h"
#include "dmc/Traits/StandardTraits.h"

#include <mlir/IR/Module.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/Analysis/Verifier.h>

#include <iostream>

using namespace mlir;
using namespace dmc;

int main() {
  MLIRContext mlirContext;
  DynamicContext ctx{&mlirContext};

  auto *dialectTest0 = ctx.createDynamicDialect("test0");
  auto *dialectTest1 = ctx.createDynamicDialect("test1");
  std::cout << "Dialects registered" << std::endl;

  auto *opA = dialectTest0->createDynamicOp("opA");
  auto *opB = dialectTest0->createDynamicOp("opB");
  auto *opC = dialectTest1->createDynamicOp("opC");
  auto *opD = dialectTest1->createDynamicOp("opD");

  opA->addOpTrait(std::make_unique<OneOperand>());
  opA->addOpTrait(std::make_unique<OneResult>());

  std::cout << "Ops registered" << std::endl;

  // Test creating modules
  ModuleWriter writer{&ctx};
  OpBuilder b{&mlirContext};

  auto testFunc = writer.createFunction("testFunc",
      {b.getIntegerType(32)}, {b.getIntegerType(64)});
  FunctionWriter funcWriter{testFunc};
  funcWriter.createOp(opA, testFunc.getArguments(), {b.getIntegerType(64)});

  writer.getModule().print(llvm::outs());
  llvm::outs() << "\n";

  if (failed(mlir::verify(writer.getModule()))) {
    llvm::outs() << "Failed to verify module\n";
  }
}
