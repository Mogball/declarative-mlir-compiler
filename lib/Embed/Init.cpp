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

static bool inited{false};

void init(MLIRContext *ctx) {
  if (inited)
    return;
  inited = true;

  setMLIRContext(ctx);
  initialize_interpreter();
}

} // end namespace py
} // end namespace mlir
