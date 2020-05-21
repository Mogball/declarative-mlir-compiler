#include "Attribute.h"
#include "Support.h"

#include <mlir/IR/Types.h>
#include <mlir/IR/Location.h>

using namespace pybind11;

namespace mlir {
namespace py {

template <typename FcnT> auto nullcheck(FcnT fcn) {
  return ::nullcheck(fcn, "attribute");
}

void exposeIntFPAttr(module &m, class_<Attribute> &attr) {
  /// TODO handle conversions from APFloat and APInt to Python floats and
  /// integers to support arbitrary-precision values.
  class_<FloatAttr>(m, "FloatAttr", attr)
      .def(init(overload<FloatAttr(Type, double)>(
           &getFloatAttr)))
      .def(init(overload<FloatAttr(Type, double, Location)>(
           &getFloatAttr)))
      .def("getValue", nullcheck(
           [](FloatAttr attr) { return attr.getValueAsDouble(); }));

  class_<IntegerAttr>(m, "IntegerAttr", attr)
      .def(init(overload<IntegerAttr(Type, int64_t)>(
           &getIntegerAttr)))
      .def(init(overload<IntegerAttr(Type, int64_t, Location)>(
           &getIntegerAttr)))
      .def("getInt", nullcheck(&IntegerAttr::getInt))
      .def("getSInt", nullcheck(&IntegerAttr::getSInt))
      .def("getUInt", nullcheck(&IntegerAttr::getUInt));
}

} // end namespace py
} // end namespace mlir
