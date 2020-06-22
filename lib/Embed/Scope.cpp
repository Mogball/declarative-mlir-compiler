#include <pybind11/pybind11.h>

using namespace pybind11;

namespace dmc {
namespace py {

module getInternalModule() {
  return module::import("mlir");
}

void ensureBuiltins(module m) {
  auto scope = m.attr("__dict__").cast<dict>();
  if (!scope.contains("__builtins__"))
    scope["__builtins__"] = PyEval_GetBuiltins();
}

} // end namespace py
} // end namespace dmc
