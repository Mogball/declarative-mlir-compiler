#include "dmc/Embed/Constraints.h"
#include "dmc/Python/PyMLIR.h"

#include <pybind11/embed.h>

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/StandardTypes.h>

using namespace mlir;
using namespace llvm;

int main() {
  MLIRContext ctx;
  py::init(&ctx);
  auto attr = IntegerAttr::get(IntegerType::get(32, &ctx), 42);
  py::evalConstraintExpr("isinstance({self}, IntegerAttr)", attr);
}
