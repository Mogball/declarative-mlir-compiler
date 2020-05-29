#include "Utility.h"
#include "Location.h"

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

using namespace llvm;
using namespace mlir;
using namespace pybind11;

namespace mlir {
namespace py {

template <typename T> auto isa() {
  return ::isa<LocationAttr, T>();
}

void exposeLocation(module &m, class_<Attribute> &attr) {
  class_<LocationAttr> locAttr{m, "LocationAttr", attr};
  locAttr
      .def(init<Location>())
      .def(self == self)
      .def(self != self)
      .def("__repr__", StringPrinter<LocationAttr>{})
      .def("__hash__", [](LocationAttr loc) { return hash_value(loc); })
      .def("isUnknownLoc", isa<UnknownLoc>())
      .def("isCallSiteLoc", isa<CallSiteLoc>())
      .def("isFileLineColLoc", isa<FileLineColLoc>())
      .def("isFusedLoc", isa<FusedLoc>())
      .def("isNameLoc", isa<NameLoc>());

  class_<Location> loc{m, "Location", locAttr};

  class_<UnknownLoc>(m, "UnknownLoc", locAttr)
      .def(init(&getUnknownLoc));

  class_<CallSiteLoc>(m, "CallSiteLoc", locAttr)
      .def(init(&getCallSiteLoc))
      .def_property_readonly("callee", &getCallee)
      .def_property_readonly("caller", &getCaller);

  class_<FileLineColLoc>(m, "FileLineColLoc", locAttr)
      .def(init(&getFileLineColLoc))
      .def_property_readonly("filename", &getFilename)
      .def_property_readonly("line", &getLine)
      .def_property_readonly("col", &getColumn);

  class_<FusedLoc>(m, "FusedLoc", locAttr)
      .def(init(&getFusedLoc))
      .def_property_readonly("locs", &getLocations);

  class_<NameLoc>(m, "NameLoc", locAttr)
      .def(init(overload<NameLoc(std::string, Location)>(&getNameLoc)))
      .def(init(overload<NameLoc(std::string)>(&getNameLoc)))
      .def_property_readonly("name", &getName)
      .def_property_readonly("child", &getChildLoc);

  implicitly_convertible<Location, LocationAttr>();
  implicitly_convertible_from_all<Location,
      LocationAttr, UnknownLoc, CallSiteLoc,
      FileLineColLoc, FusedLoc, NameLoc>(loc);
}

} // end namespace py
} // end namespace mlir
