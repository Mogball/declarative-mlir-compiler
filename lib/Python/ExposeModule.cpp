#include "Module.h"
#include "Support.h"
#include "OwningModuleRef.h"

#include <pybind11/pybind11.h>

using namespace pybind11;

namespace mlir {
namespace py {

void exposeModule(module &m) {
  class_<OwningModuleRef>(m, "OwningModuleRef")
      .def(init<>())
      .def("get", &getOwnedModule)
      .def("release", &OwningModuleRef::release)
      .def("__repr__", &printModuleRef)
      .def("__bool__", &OwningModuleRef::operator bool);
  class_<ModuleOp>(m, "ModuleOp")
      .def(init<>())
      .def(init<const ModuleOp &>())
      .def("__repr__", nullcheck(StringPrinter<ModuleOp>{}, "ModuleOp"))
      .def_property_readonly("name", &getName);
  m.def("Module", overload<ModuleOp()>(&getModuleOp));
  m.def("Module", overload<ModuleOp(Location)>(&getModuleOp));
  m.def("Module", overload<ModuleOp(std::string)>(&getModuleOp));
  m.def("Module", overload<ModuleOp(Location, std::string)>(&getModuleOp));
}

} // end namespace py
} // end namespace mlir
