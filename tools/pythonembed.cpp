#include "dmc/Embed/Init.h"
#include "dmc/Embed/Constraints.h"
#include "dmc/Embed/OpFormatGen.h"
#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/Traits/Registry.h"
#include "dmc/Python/PyMLIR.h"
#include "dmc/IO/ModuleWriter.h"
#include "dmc/Spec/SpecDialect.h"

#include <pybind11/embed.h>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Builders.h>

using namespace mlir;
using namespace llvm;
using namespace ::dmc;
using namespace pybind11;

class TestOp : public Op<TestOp, mlir::OpTrait::AtLeastNOperands<1>::Impl> {
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
  DialectRegistration<SpecDialect>{};
  DialectRegistration<TraitRegistry>{};
  MLIRContext ctx;
  DynamicContext dynCtx{&ctx};
  OpBuilder b{&ctx};

  StringRef name{"fmt_op"};
  auto loc = b.getUnknownLoc();
  auto opType = OpType::getChecked(loc, {
    {"temp", b.getIntegerType(32)},
  }, {
    {"res", b.getIntegerType(64)},
  });
  std::vector<mlir::NamedAttribute> opAttrs;
  opAttrs.push_back({b.getIdentifier("offset"),
                    b.getStringAttr("xd")});
  auto opRegion = OpRegion::getChecked(loc, {});
  auto opSucc = OpSuccessor::getChecked(loc, {});
  auto opTraits = OpTraitsAttr::getChecked(loc, b.getArrayAttr({}));

  ModuleWriter mw{&dynCtx};
  auto fcn = mw.createFunction("testFcn", {b.getIntegerType(32)}, {});
  FunctionWriter fw{fcn};
  b.setInsertionPointToStart(&fcn.front());
  std::vector<mlir::NamedAttribute> attrs;
  attrs.push_back({b.getIdentifier("fmt"),
      b.getStringAttr("$temp $offset attr-dict-with-keyword `>` functional-type($temp, $res)")});
  auto opOp = b.create<OperationOp>(loc, name, opType, opAttrs, opRegion, opSucc, opTraits, attrs);


  std::string parserStr, printerStr;
  llvm::raw_string_ostream parser{parserStr}, printer{printerStr};
  if (failed(generateOpFormat(opOp, parser, printer)))
    errs() << "Failed to generate op format\n";

  errs() << "\nParser:\n";
  errs() << parser.str() << "\n";
  errs() << "\nPrinter:\n";
  errs() << printer.str() << "\n";
  /*
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
  */
}
