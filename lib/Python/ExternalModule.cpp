#include "dmc/Python/PyMLIR.h"
#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/Dynamic/DynamicDialect.h"
#include "dmc/Spec/DialectGen.h"
#include "dmc/Spec/SpecOps.h"
#include "dmc/Embed/Expose.h"

#include <pybind11/embed.h>

using namespace dmc;
using namespace mlir;
using namespace pybind11;

PYBIND11_MODULE(mlir, m) {
  mlir::py::getModule(m);
  // ownership is given to MLIRContext
  auto *ctx = new DynamicContext{mlir::py::getMLIRContext()};

  m.def("registerDynamicDialects", [ctx](ModuleOp module) {
    list ret;
    for (auto dialectOp : module.getOps<DialectOp>()) {
      if (failed(registerDialect(dialectOp, ctx)))
        throw std::invalid_argument{"Failed to register dialect: " +
                                    dialectOp.getName().str()};
      auto *dialect =
          mlir::py::getMLIRContext()->getRegisteredDialect(dialectOp.getName());
      dmc::py::exposeDialectInternal(dynamic_cast<DynamicDialect *>(dialect));
      ret.append(eval(dialect->getNamespace().str(),
                      module::import("mlir").attr("__dict__")));
    }
    return ret;
  });
}
