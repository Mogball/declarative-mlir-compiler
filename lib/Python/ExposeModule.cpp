#include "Expose.h"
#include "Location.h"
#include "Module.h"
#include "Utility.h"
#include "Identifier.h"
#include "OwningModuleRef.h"

#include <pybind11/pybind11.h>

using namespace pybind11;

namespace mlir {
namespace py {

template <typename FcnT> auto nullcheck(FcnT fcn) {
  return ::nullcheck(fcn, "module");
}

FuncOp functionCtor(std::string name, FunctionType ty, Location loc,
                    AttrDictRef attrDict) {
  NamedAttrList attrs;
  for (auto &[name, attr] : attrDict)
    attrs.push_back({getIdentifierChecked(name), attr});
  return FuncOp::create(loc, name, ty, ArrayRef<NamedAttribute>{attrs});
}

void exposeModule(module &m, OpClass &cls) {
  class_<ModuleOp>(m, "ModuleOp", cls)
      .def(init([](Location loc) { return ModuleOp::create(loc); }),
           "location"_a = getUnknownLoc())
      .def(init([](std::string name, Location loc) {
        return ModuleOp::create(loc, StringRef{name});
      }), "name"_a, "location"_a = getUnknownLoc())
      .def("__repr__", nullcheck(StringPrinter<ModuleOp>{}))
      .def("__bool__", &ModuleOp::operator bool)
      .def_property_readonly("name", nullcheck(&getModuleName))
      .def_property_readonly("region", nullcheck(&ModuleOp::getBodyRegion),
                             return_value_policy::reference)
      .def_property_readonly("body", nullcheck(&ModuleOp::getBody),
                             return_value_policy::reference)
      .def("__iter__", nullcheck([](ModuleOp module) {
        return make_iterator(module.begin(), module.end());
      }), keep_alive<0, 1>())
      .def("append", nullcheck(&ModuleOp::push_back), "op"_a)
      .def("insertBefore",
           nullcheck(overload<void(ModuleOp::*)(Operation *, Operation *)>(
               &ModuleOp::insert)),
           "insertPt"_a, "op"_a);

  class_<FuncOp>(m, "FuncOp", cls)
      .def(init(&functionCtor), "name"_a, "type"_a, "loc"_a = getUnknownLoc(),
           "attrs"_a = AttrDict{})
      .def("addEntryBlock", &FuncOp::addEntryBlock,
           return_value_policy::reference)
      .def("addBlock", &FuncOp::addBlock, return_value_policy::reference);
}

} // end namespace py
} // end namespace mlir
