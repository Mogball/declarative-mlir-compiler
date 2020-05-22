#include "Attribute.h"
#include "Support.h"
#include "Location.h"

#include <mlir/IR/Types.h>

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
      .def(init(&getFloatAttr), "type"_a, "value"_a,
                                "location"_a = getUnknownLoc())
      .def("getValue", nullcheck(
           [](FloatAttr attr) { return attr.getValueAsDouble(); }));

  class_<IntegerAttr>(m, "IntegerAttr", attr)
      .def(init(&getIntegerAttr), "type"_a, "value"_a,
                                  "location"_a = getUnknownLoc())
      .def("getInt", nullcheck(&IntegerAttr::getInt))
      .def("getSInt", nullcheck(&IntegerAttr::getSInt))
      .def("getUInt", nullcheck(&IntegerAttr::getUInt));
}

} // end namespace py
} // end namespace mlir
