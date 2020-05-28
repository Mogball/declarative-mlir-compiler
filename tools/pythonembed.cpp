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

  auto arg = IntegerType::get(32, &ctx);
  StringRef expr{"isinstance({self}, IntegerType)"};
  auto name = "type";

  /// Substitute placeholder `self -> name`.
  dict fmtArgs{"self"_a = name};
  auto pyExpr = pybind11::cast(expr.str()).cast<str>().format(**fmtArgs);
  /// Wrap in a function.
  auto scope = module::import("__main__").attr("__dict__");
  dict funcExpr{"name"_a = name, "expr"_a = pyExpr};
  auto funcStr = "def constraint({name}): return {expr}"_s.format(**funcExpr);
  exec(funcStr, scope);


  print(funcStr);
  print(arg);
  print(scope["constraint"](arg));


  py::evalConstraintExpr(expr, arg);
}
