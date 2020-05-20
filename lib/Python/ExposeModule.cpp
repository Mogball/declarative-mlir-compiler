#include "Module.h"
#include "Support.h"
#include "OwningModuleRef.h"

#include <boost/python.hpp>

namespace mlir {
namespace py {

void exposeModule() {
  using namespace boost;
  using namespace boost::python;
  class_<OwningModuleRef, noncopyable>("OwningModuleRef", no_init)
      .def(self_ns::repr(self_ns::self))
      .def("get", &getOwnedModule)
      .def("release", &OwningModuleRef::release)
      .def("__bool__", &OwningModuleRef::operator bool);
  class_<ModuleOp>("ModuleOp", no_init)
      .def(self_ns::repr(self_ns::self))
      .add_property("name", &getName);
  def("Module", overload<ModuleOp()>(&getModuleOp));
  def("Module", overload<ModuleOp(Location)>(&getModuleOp));
  def("Module", overload<ModuleOp(std::string)>(&getModuleOp));
  def("Module", overload<ModuleOp(Location, std::string)>(&getModuleOp));
}

} // end namespace py
} // end namespace mlir
