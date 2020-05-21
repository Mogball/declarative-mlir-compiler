#include "Support.h"
#include "Context.h"

#include <mlir/IR/Identifier.h>
#include <mlir/IR/Attributes.h>

using namespace pybind11;

namespace mlir {
namespace py {

template <typename FcnT> auto nullcheck(FcnT fcn) {
  return ::nullcheck(fcn, "symbol reference attribute");
}

void exposeSymbolRefAttr(module &m, class_<Attribute> &attr) {
  class_<SymbolRefAttr> symbolRefAttr{m, "SymbolRefAttr", attr};
  symbolRefAttr
      .def(init([](const std::string &value,
                   const std::vector<FlatSymbolRefAttr> &refs) {
        return SymbolRefAttr::get(value, refs, getMLIRContext());
      }))
      .def_property_readonly("root", nullcheck([](SymbolRefAttr attr)
                             { return attr.getRootReference().str(); }))
      .def_property_readonly("leaf", nullcheck([](SymbolRefAttr attr)
                             { return attr.getLeafReference().str(); }))
      .def("getNestedReferences", nullcheck([](SymbolRefAttr attr) {
        auto refs = attr.getNestedReferences();
        return std::vector<FlatSymbolRefAttr>{std::begin(refs),
                                              std::end(refs)};
      }));

  // FlatSymbolRefAttr subclasses SymbolRefAttr
  class_<FlatSymbolRefAttr>(m, "FlatSymbolRefAttr", symbolRefAttr)
      .def(init([](const std::string &value) {
        return FlatSymbolRefAttr::get(value, getMLIRContext());
      }))
      .def("getValue", nullcheck([](FlatSymbolRefAttr attr) {
        return attr.getValue().str();
      }));
}

} // end namespace py
} // end namespace mlir
