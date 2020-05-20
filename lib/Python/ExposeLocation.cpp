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
      .def("isUnknownLoc", &isUnknownLoc)
      .def("isCallSiteLoc", &isCallSiteLoc)
      .add_property("callee", &getCallee)
      .add_property("caller", &getCaller)
      .def("isFileLineColLoc", &isFileLineColLoc)
      .add_property("filename", &getFilename)
      .add_property("line", &getLine)
      .add_property("col", &getColumn)
      .def("isFusedLoc", &isFusedLoc)
      .add_property("locs", &getLocations)
      .def("isNameLoc", &isNameLoc)
      .add_property("name", &getName)
      .add_property("child", &getChildLoc);
  /// Getters.
  def("UnknownLoc", &getUnknownLoc);
  def("CallSiteLoc",
      overload<Location(Location, Location)>(&CallSiteLoc::get));
  def("FileLineColLoc", &getFileLineColLoc);
  def("FusedLoc", &getFusedLoc);
  def("NameLoc", overload<Location(std::string, Location)>(&getNameLoc));
  def("NameLoc", overload<Location(std::string)>(&getNameLoc));
}

} // end namespace py
} // end namespace mlir
