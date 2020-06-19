#include "dmc/Python/PyMLIR.h"
#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/Spec/DialectGen.h"

using namespace dmc;
using namespace mlir;

PYBIND11_MODULE(mlir, m) {
  py::getModule(m);
  // ownership is given to MLIRContext
  auto *ctx = new DynamicContext{mlir::py::getMLIRContext()};

  m.def("registerDynamicDialects", [ctx](ModuleOp module) {
    if (failed(registerAllDialects(module, ctx)))
      throw std::invalid_argument{"Failed to register dynamic dialects"};
  });
}
