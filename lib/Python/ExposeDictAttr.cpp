#include "Attribute.h"
#include "Support.h"

#include <mlir/IR/Identifier.h>
#include <pybind11/stl.h>

using namespace pybind11;

namespace mlir {
namespace py {

template <typename FcnT> auto nullcheck(FcnT fcn) {
  return ::nullcheck(fcn, "dictionary attribute");
}

void exposeDictAttr(module &m, class_<Attribute> &attr) {
  class_<DictionaryAttr>(m, "DictionaryAttr", attr)
      .def(init(&getDictionaryAttr))
      .def("getValue", nullcheck(&getDictionaryAttrValue))
      .def("empty", nullcheck([](DictionaryAttr attr) { return attr.empty(); }))
      .def("__iter__", nullcheck([](DictionaryAttr attr) {
        return make_key_iterator(attr.begin(), attr.end());
      }), keep_alive<0, 1>())
      .def("items", nullcheck([](DictionaryAttr attr) {
        return make_iterator(attr.begin(), attr.end());
      }), keep_alive<0, 1>())
      .def("__getitem__", nullcheck(&dictionaryAttrGetItem))
      .def("__contains__", nullcheck(
           [](DictionaryAttr attr, const std::string &key) -> bool {
             return !!attr.get(key);
           }))
      .def("__len__", nullcheck(
           [](DictionaryAttr attr) { return attr.size(); }));
}

} // end namespace py
} // end namespace mlir
