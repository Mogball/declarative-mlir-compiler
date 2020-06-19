#include <pybind11/pybind11.h>

using namespace pybind11;

namespace dmc {
namespace py {

module getInternalModule() {
  return module::import("mlir");
}

} // end namespace py
} // end namespace dmc
