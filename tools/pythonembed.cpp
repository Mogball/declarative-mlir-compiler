#include "dmc/Embed/Init.h"
#include "dmc/Embed/Constraints.h"
#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/Python/PyMLIR.h"

#include <pybind11/embed.h>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/StandardTypes.h>

using namespace mlir;
using namespace llvm;
using namespace dmc;
using namespace pybind11;


int main() {
  MLIRContext ctx;
  DynamicContext dynCtx{&ctx};

  StringRef expr = "isinstance({self}, IntegerType)";
  auto arg = IntegerType::get(32, &ctx);

  errs() << (succeeded(py::evalConstraintExpr(expr, (Type)arg))
             ? "success" : "failure") << "\n";

  errs() << (succeeded(py::evalConstraintExpr(expr, arg))
             ? "success" : "failure") << "\n";

}
