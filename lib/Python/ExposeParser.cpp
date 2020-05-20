#include "Module.h"
#include "Parser.h"
#include "OwningModuleRef.h"

#include <boost/python.hpp>

using namespace mlir;
using namespace llvm;

namespace mlir {
namespace py {

void exposeParser() {
  using namespace boost;
  using namespace boost::python;
  class_<OwningModuleRef, noncopyable>("OwningModuleRef", no_init)
      .def(self_ns::repr(self_ns::self));
  class_<ModuleOp>("ModuleOp", no_init)
      .def(self_ns::repr(self_ns::self));
  def("parseSourceFile", parseSourceFile,
      return_value_policy<manage_new_object>{});
}

} // end namespace py
} // end namespace mlir
