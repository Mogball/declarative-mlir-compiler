#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/IO/ModuleWriter.h"
#include "dmc/Traits/StandardTraits.h"
#include "dmc/Spec/SpecDialect.h"
#include "dmc/Spec/SpecTypes.h"

#include <mlir/IR/Module.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/Analysis/Verifier.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include <iostream>

using namespace mlir;
using namespace dmc;

static DialectRegistration<StandardOpsDialect> registerStandardOps;
static DialectRegistration<SpecDialect> registerSpecOps;

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
  opA->addOpTrait(std::make_unique<IsCommutative>());

  // Finalize ops
  opA->finalize();
  opB->finalize();
  opC->finalize();
  opD->finalize();

  std::cout << "Ops registered" << std::endl;

  // Test creating modules
  ModuleWriter writer{&ctx};
  OpBuilder b{&mlirContext};

  auto testFunc = writer.createFunction("testFunc",
      {b.getIntegerType(32)}, {b.getIntegerType(64)});
  FunctionWriter funcWriter{testFunc};
  auto *testOpInstance = funcWriter.createOp(opA, testFunc.getArguments(), {b.getIntegerType(64)});
  assert(testOpInstance->isCommutative());
  funcWriter.createOp("std.return", testOpInstance->getResult(0), {});

  writer.getModule().print(llvm::outs());
  llvm::outs() << "\n";

  if (failed(mlir::verify(writer.getModule()))) {
    llvm::outs() << "Failed to verify module\n";
  }

  // Check AnyOf is order-independent
  auto anyOf0 = AnyOfType::get({b.getIntegerType(16), b.getIntegerType(32), b.getIntegerType(64)});
  auto anyOf1 = AnyOfType::get({b.getIntegerType(64), b.getIntegerType(16), b.getIntegerType(32)});
  auto anyOf2 = AnyOfType::get({b.getIntegerType(64), b.getIntegerType(32), b.getIntegerType(16)});
  assert(anyOf0.getAsOpaquePointer() == anyOf1.getAsOpaquePointer());
  assert(anyOf2.getAsOpaquePointer() == anyOf1.getAsOpaquePointer());
  auto anyOf3 = AnyOfType::get({b.getIntegerType(64), b.getIntegerType(32), b.getIntegerType(8)});
  assert(anyOf3.getAsOpaquePointer() != anyOf0.getAsOpaquePointer());
}
