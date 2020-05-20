#include "Support.h"
#include "Location.h"

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

using namespace llvm;
using namespace pybind11;

namespace mlir {
namespace py {

void exposeLocation(module &m) {
  class_<Location>(m, "Location")
      .def(init<const Location &>())
      .def(self == self)
      .def(self != self)
      .def("__repr__", StringPrinter<Location>{})
      .def("__hash__", overload<hash_code(Location)>(&hash_value))
      .def("isUnknownLoc", &isUnknownLoc)
      .def("isCallSiteLoc", &isCallSiteLoc)
      .def_property_readonly("callee", &getCallee)
      .def_property_readonly("caller", &getCaller)
      .def("isFileLineColLoc", &isFileLineColLoc)
      .def_property_readonly("filename", &getFilename)
      .def_property_readonly("line", &getLine)
      .def_property_readonly("col", &getColumn)
      .def("isFusedLoc", &isFusedLoc)
      .def_property_readonly("locs", &getLocations)
      .def("isNameLoc", &isNameLoc)
      .def_property_readonly("name", &getName)
      .def_property_readonly("child", &getChildLoc);
  /// Getters.
  m.def("UnknownLoc", &getUnknownLoc);
  m.def("CallSiteLoc",
      overload<Location(Location, Location)>(&CallSiteLoc::get));
  m.def("FileLineColLoc", &getFileLineColLoc);
  m.def("FusedLoc", &getFusedLoc);
  m.def("NameLoc", overload<Location(std::string, Location)>(&getNameLoc));
  m.def("NameLoc", overload<Location(std::string)>(&getNameLoc));
}

} // end namespace py
} // end namespace mlir
