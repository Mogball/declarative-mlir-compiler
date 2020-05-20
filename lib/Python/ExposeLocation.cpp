#include "Support.h"
#include "Location.h"

#include <boost/python.hpp>

using namespace llvm;

namespace mlir {
namespace py {

void exposeLocation() {
  using namespace boost;
  using namespace boost::python;
  class_<Location>("Location", no_init)
      .def(self_ns::repr(self_ns::self))
      .def(self == self)
      .def(self != self)
      .def("__hash__", overload<hash_code(Location)>(&hash_value))
      .def("isCallSiteLoc", &isCallSiteLoc)
      .def("getCallee", &getCallee)
      .def("getCaller", &getCaller)
      .def("isFileLineColLoc", &isFileLineColLoc)
      .def("getFilename", &getFilename)
      .def("getLine", &getLine)
      .def("getColumn", &getColumn);
  /// Getters.
  def("getUnknownLoc", &getUnknownLoc);
  def("getCallSiteLoc",
      overload<Location(Location, Location)>(&CallSiteLoc::get));
  def("getFileLineColLoc", &getFileLineColLoc);
}

} // end namespace py
} // end namespace mlir
