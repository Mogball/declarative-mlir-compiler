#include "Expose.h"
#include "Location.h"
#include "Module.h"
#include "Utility.h"
#include "OwningModuleRef.h"

#include <pybind11/pybind11.h>

using namespace pybind11;

namespace mlir {
namespace py {

template <typename FcnT> auto nullcheck(FcnT fcn) {
  return ::nullcheck(fcn, "module");
}

void exposeModule(module &m, OpClass &cls) {
  class_<ModuleOp>(m, "ModuleOp", cls)
      .def(init([](Location loc) { return ModuleOp::create(loc); }),
           "location"_a = getUnknownLoc())
      .def(init([](std::string name, Location loc) {
        return ModuleOp::create(loc, StringRef{name});
      }), "name"_a, "location"_a = getUnknownLoc())
      .def("__repr__", nullcheck(StringPrinter<ModuleOp>{}))
      .def("__bool__", &ModuleOp::operator bool)
      .def_property_readonly("name", nullcheck(&getModuleName));
}

} // end namespace py
} // end namespace mlir
