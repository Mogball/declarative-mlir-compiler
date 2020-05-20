#include "Parser.h"

#include <pybind11/pybind11.h>

using namespace pybind11;

namespace mlir {
namespace py {

void exposeParser(module &m) {
  m.def("parseSourceFile", &parseSourceFile);
}

} // end namespace py
} // end namespace mlir
