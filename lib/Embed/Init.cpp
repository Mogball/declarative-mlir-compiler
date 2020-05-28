#include "Scope.h"
#include "dmc/Embed/Constraints.h"
#include "dmc/Python/PyMLIR.h"

#include <pybind11/embed.h>

namespace {
PYBIND11_EMBEDDED_MODULE(mlir, m) {
  mlir::py::getModule(m);
}
} // end anonymous namespace

using namespace pybind11;

namespace mlir {
namespace py {

void init(MLIRContext *ctx) {
  setMLIRContext(ctx);
  initialize_interpreter();
  /// Add pymlir's objects to the main scope.
  exec("from mlir import *", getMainScope());
}

} // end namespace py
} // end namespace mlir
