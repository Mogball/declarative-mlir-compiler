#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/IO/ModuleWriter.h"
#include "dmc/Traits/StandardTraits.h"
#include "dmc/Traits/Registry.h"
#include "dmc/Spec/SpecDialect.h"
#include "dmc/Spec/SpecTypes.h"
#include "dmc/Spec/SpecAttrs.h"

#include <mlir/Parser.h>
#include <mlir/IR/Module.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>

#include <iostream>

using namespace mlir;
using namespace dmc;

static DialectRegistration<StandardOpsDialect> registerStandardOps;
static DialectRegistration<SpecDialect> registerSpecOps;
static DialectRegistration<TraitRegistry> registerTraits;

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

  opA->addOpTrait<NOperands>(1);
  opA->addOpTrait<NResults>(1);
  opA->addOpTrait<IsCommutative>();

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
  {
    auto anyOf0 = AnyOfType::get({b.getIntegerType(16), b.getIntegerType(32), b.getIntegerType(64)});
    auto anyOf1 = AnyOfType::get({b.getIntegerType(64), b.getIntegerType(16), b.getIntegerType(32)});
    auto anyOf2 = AnyOfType::get({b.getIntegerType(64), b.getIntegerType(32), b.getIntegerType(16)});
    assert(anyOf0.getAsOpaquePointer() == anyOf1.getAsOpaquePointer());
    assert(anyOf2.getAsOpaquePointer() == anyOf1.getAsOpaquePointer());
    auto anyOf3 = AnyOfType::get({b.getIntegerType(64), b.getIntegerType(32), b.getIntegerType(8)});
    assert(anyOf3.getAsOpaquePointer() != anyOf0.getAsOpaquePointer());
  }

  {
    auto anyWidth = AnyIntOfWidthsType::get({8, 16, 32}, &mlirContext);
    assert(failed(anyWidth.verify(b.getIntegerType(64))));
    assert(succeeded(anyWidth.verify(b.getIntegerType(32))));
  }

  {
    auto complexTy = dmc::ComplexType::get(AnyIntegerType::get(&mlirContext));
    auto complexTyArg0 = mlir::ComplexType::get(b.getIntegerType(64));
    auto complexTyArg1 = mlir::ComplexType::get(b.getIntegerType(16));
    auto complexTyArg2 = mlir::ComplexType::get(b.getF32Type());
    assert(succeeded(complexTy.verify(complexTyArg0)));
    assert(succeeded(complexTy.verify(complexTyArg1)));
    assert(failed(complexTy.verify(complexTyArg2)));
    assert(failed(complexTy.verify(b.getIntegerType(64))));
    auto complexTy0 = dmc::ComplexType::get(b.getIntegerType(64));
    assert(succeeded(complexTy0.verify(complexTyArg0)));
    assert(failed(complexTy0.verify(complexTyArg1)));
    assert(failed(complexTy0.verify(complexTyArg2)));
  }

  {
    auto anyOfCompound = AnyOfType::get({
        AnyIntegerType::get(&mlirContext), b.getF32Type()});
    assert(succeeded(anyOfCompound.verify(b.getIntegerType(32))));
    assert(succeeded(anyOfCompound.verify(b.getIntegerType(16))));
    assert(succeeded(anyOfCompound.verify(b.getF32Type())));
    assert(failed(anyOfCompound.verify(b.getF16Type())));
    auto anyOfNested = AnyOfType::get({AnyOfType::get({AnyFloatType::get(&mlirContext)})});
    assert(succeeded(anyOfNested.verify(b.getF16Type())));
    assert(failed(anyOfNested.verify(b.getIntegerType(32))));
  }

  {
    auto anyI = AnyIAttr::get(AnyIType::get(32, &mlirContext));
    assert(succeeded(anyI.verify(b.getI32IntegerAttr(1))));
    assert(succeeded(anyI.verify(b.getI32IntegerAttr(2))));
    assert(failed(anyI.verify(b.getI64IntegerAttr(1))));
    assert(failed(anyI.verify(b.getIndexAttr(1))));
    assert(failed(anyI.verify(b.getStringAttr("hello"))));
    auto anyI0 = AnyIAttr::get(AnyIType::get(64, &mlirContext));
    assert(succeeded(anyI0.verify(b.getI64IntegerAttr(1))));
    assert(failed(anyI0.verify(b.getI32IntegerAttr(2))));
    assert(failed(anyI0.verify(b.getStringAttr("hello"))));
  }

  {
    std::vector<Type> inputs = {
        AnyOfType::get({AnyIntegerType::get(&mlirContext), b.getF32Type()}),
        AnyOfType::get({AnyFloatType::get(&mlirContext), b.getIntegerType(64)})
    };
    std::vector<Type> outputs = {
        AnyIType::get(16, &mlirContext)
    };
    auto opTy = mlir::FunctionType::get(inputs, outputs, &mlirContext);
    opTy.print(llvm::outs());
    llvm::outs() << "\n";

    auto module = ModuleOp::create(UnknownLoc::get(&mlirContext));
    std::vector<Type> inTys = {
      b.getF32Type(),
      b.getIntegerType(64),
    };
    std::vector<Type> outTys = {
      b.getIntegerType(32),
    };
    auto funcTy = mlir::FunctionType::get(inTys, outTys, &mlirContext);
    auto func = FuncOp::create(UnknownLoc::get(&mlirContext), "test", funcTy);
    module.push_back(func);
    auto *entry = func.addEntryBlock();
    OpBuilder builder{&mlirContext};
    builder.setInsertionPointToStart(entry);

    OperationState state{UnknownLoc::get(&mlirContext),
        {"test0.opA", &mlirContext}};
    state.addTypes({b.getIntegerType(16)});
    state.addOperands(entry->getArguments());
    state.addAttribute("myAttr", b.getI32IntegerAttr(42));
    auto *op = builder.createOperation(state);

    auto ret = builder.create<ReturnOp>(UnknownLoc::get(&mlirContext));

    auto myAttr = AnyIAttr::get(AnyIType::get(32, &mlirContext));
    std::vector<NamedAttribute> opAttrs{b.getNamedAttr("myAttr", myAttr)};

    dmc::impl::verifyTypeConstraints(op, opTy);
    dmc::impl::verifyAttrConstraints(op, b.getDictionaryAttr(opAttrs));

    module.print(llvm::outs());
    llvm::outs() << "\n";
  }

  llvm::errs() << mlir::parseType("!dmc.Any", &mlirContext) << "\n";
}
