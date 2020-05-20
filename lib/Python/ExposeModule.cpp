#include "Module.h"
#include "OwningModuleRef.h"

#include <boost/python.hpp>

namespace mlir {
namespace py {

void exposeModule() {
  using namespace boost;
  using namespace boost::python;
  class_<OwningModuleRef, noncopyable>("OwningModuleRef", no_init)
      .def(self_ns::repr(self_ns::self));
  class_<ModuleOp>("ModuleOp", no_init)
      .def(self_ns::repr(self_ns::self));
}

} // end namespace py
} // end namespace mlir
