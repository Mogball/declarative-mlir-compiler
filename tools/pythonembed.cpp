#include "dmc/Embed/Init.h"
#include "dmc/Embed/Constraints.h"
#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/Python/PyMLIR.h"
#include "dmc/IO/ModuleWriter.h"

#include <pybind11/embed.h>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>

using namespace mlir;
using namespace llvm;
using namespace dmc;
using namespace pybind11;

class TestOp : public Op<TestOp, OpTrait::AtLeastNOperands<1>::Impl> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "test.op"; }

  static void build(OpBuilder &builder, OperationState &result,
                    ValueRange operands, TypeRange results,
                    NamedAttrList attrs) {
    result.addOperands(operands);
    result.addTypes(results);
    result.attributes.append(attrs);
  }

  void print(OpAsmPrinter &printer) {
    auto scope = module::import("__main__").attr("__dict__");
    scope["print_test_op"].operator()<return_value_policy::reference>(
      printer, getOperation());
  }
};

class TestDialect : public Dialect {
public:
  TestDialect(MLIRContext *ctx) : Dialect("test", ctx) {
    addOperations<TestOp>();
  }
};

int main() {
  DialectRegistration<TestDialect>{};
  MLIRContext ctx;
  DynamicContext dynCtx{&ctx};

  auto scope = module::import("__main__").attr("__dict__");
  exec(R"(
    def print_test_op(printer, op):
      printer.print(op.name, '(', op.getOperands(), ')')
  )", scope);

  ModuleWriter mw{&dynCtx};
  OpBuilder b{&ctx};
  auto fcn = mw.createFunction("testFcn", {b.getIntegerType(32)}, {});
  FunctionWriter fw{fcn};
  b.setInsertionPointToStart(&fcn.front());
  b.create<TestOp>(b.getUnknownLoc(), fcn.getArguments(),
                   llvm::makeArrayRef((Type)b.getF64Type()), NamedAttrList{});

  mw.getModule().print(outs());
  outs() << "\n";
}
