#include "Attribute.h"
#include "Utility.h"

#include <pybind11/stl.h>

using namespace pybind11;

namespace mlir {
namespace py {

template <typename FcnT> auto nullcheck(FcnT fcn) {
  return ::nullcheck(fcn, "array attribute");
}

void exposeArrayAttr(module &m, class_<Attribute> &attr) {
  class_<ArrayAttr>(m, "ArrayAttr", attr)
      .def(init(&getArrayAttr))
      .def("getValue", nullcheck(&getArrayAttrValue))
      .def("empty", nullcheck([](ArrayAttr attr) { return attr.size() == 0; }))
      .def("__getitem__", nullcheck(&arrayGetSlice))
      .def("__getitem__", nullcheck(&arrayGetIndex))
      .def("__len__", nullcheck([](ArrayAttr attr) { return attr.size(); }))
      .def("__iter__", nullcheck([](ArrayAttr attr) {
        return make_iterator(attr.begin(), attr.end());
      }), keep_alive<0, 1>());
}

} // end namespace py
} // end namespace mlir
