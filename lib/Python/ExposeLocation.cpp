#include "Support.h"
#include "Location.h"

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

using namespace llvm;
using namespace mlir;
using namespace pybind11;

namespace pybind11 {
template<> struct polymorphic_type_hook<Location> {
  static const void *get(const Location *src, const std::type_info *&type) {
    if (src->isa<UnknownLoc>()) {
      type = &typeid(UnknownLoc);
      return static_cast<const UnknownLoc *>(src);
    } else if (src->isa<CallSiteLoc>()) {
      type = &typeid(CallSiteLoc);
      return static_cast<const CallSiteLoc *>(src);
    } else if (src->isa<FileLineColLoc>()) {
      type = &typeid(FileLineColLoc);
      return static_cast<const FileLineColLoc *>(src);
    } else if (src->isa<FusedLoc>()) {
      type = &typeid(FusedLoc);
      return static_cast<const FusedLoc *>(src);
    } else if (src->isa<NameLoc>()) {
      type = &typeid(NameLoc);
      return static_cast<const NameLoc *>(src);
    }
    return src;
  }
};
} // end namespace pybind11

namespace mlir {
namespace py {

void exposeLocation(module &m) {
  class_<Location> locCls{m, "Location"};
  locCls
      .def(init<const Location &>())
      .def(self == self)
      .def(self != self)
      .def("__repr__", StringPrinter<Location>{})
      .def("__hash__", overload<hash_code(Location)>(&hash_value))
      .def("isUnknownLoc", &isUnknownLoc)
      .def("isCallSiteLoc", &isCallSiteLoc)
      .def("isFileLineColLoc", &isFileLineColLoc)
      .def("isFusedLoc", &isFusedLoc)
      .def("isNameLoc", &isNameLoc);

  class_<UnknownLoc>(m, "UnknownLoc", locCls)
      .def(init(&getUnknownLoc));

  class_<CallSiteLoc>(m, "CallSiteLoc", locCls)
      .def(init(&getCallSiteLoc))
      .def_property_readonly("callee", &getCallee)
      .def_property_readonly("caller", &getCaller);

  class_<FileLineColLoc>(m, "FileLineColLoc", locCls)
      .def(init(&getFileLineColLoc))
      .def_property_readonly("filename", &getFilename)
      .def_property_readonly("line", &getLine)
      .def_property_readonly("col", &getColumn);

  class_<FusedLoc>(m, "FusedLoc", locCls)
      .def(init(&getFusedLoc))
      .def_property_readonly("locs", &getLocations);

  class_<NameLoc>(m, "NameLoc", locCls)
      .def(init(overload<NameLoc(std::string, Location)>(&getNameLoc)))
      .def(init(overload<NameLoc(std::string)>(&getNameLoc)))
      .def_property_readonly("name", &getName)
      .def_property_readonly("child", &getChildLoc);
}

} // end namespace py
} // end namespace mlir
