#include "Expose.h"
#include "Context.h"
#include "Location.h"
#include "Module.h"
#include "Utility.h"
#include "Identifier.h"
#include "OwningModuleRef.h"

#include <mlir/Dialect/StandardOps/IR/Ops.h>

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

ReturnOp returnCtor(ValueListRef operands, Location loc) {
  OpBuilder b{getMLIRContext()};
  return b.create<ReturnOp>(loc, operands);
}

ConstantOp constantCtor(Attribute value, Location loc) {
  OpBuilder b{getMLIRContext()};
  return b.create<ConstantOp>(loc, value);
}

CallIndirectOp callIndirectCtor(Value callee, ValueListRef operands,
                                Location loc) {
  OpBuilder b{getMLIRContext()};
  return b.create<CallIndirectOp>(loc, callee, operands);
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
      .def("getOps", nullcheck([](ModuleOp module, object cls) {
        auto name = cls.attr("getName")().cast<std::string>();
        auto ops = llvm::make_filter_range(module, [name](Operation &op) {
          return op.getName().getStringRef() == name;
        });
        return make_iterator(ops.begin(), ops.end());
      }), keep_alive<0, 1>())
      .def("append", nullcheck(&ModuleOp::push_back), "op"_a)
      .def("insertBefore",
           nullcheck(overload<void(ModuleOp::*)(Operation *, Operation *)>(
               &ModuleOp::insert)),
           "insertPt"_a, "op"_a);

  class_<FuncOp>(m, "FuncOp", cls)
      .def(init(&functionCtor), "name"_a, "type"_a, "loc"_a = getUnknownLoc(),
           "attrs"_a = AttrDict{})
      .def_static("getName", []() { return FuncOp::getOperationName().str(); })
      .def("addEntryBlock", &FuncOp::addEntryBlock,
           return_value_policy::reference)
      .def("addBlock", &FuncOp::addBlock, return_value_policy::reference)
      .def("getBody", &FuncOp::getBody, return_value_policy::reference)
      .def("clone", overload<FuncOp(FuncOp::*)()>(&FuncOp::clone));

  class_<ReturnOp>(m, "ReturnOp", cls)
      .def(init(&returnCtor), "operands"_a = ValueList{},
           "loc"_a = getUnknownLoc());

  class_<ConstantOp>(m, "ConstantOp", cls)
      .def(init(&constantCtor), "value"_a, "loc"_a = getUnknownLoc())
      .def_static("getName",
                  []() { return ConstantOp::getOperationName().str(); })
      .def("value", &ConstantOp::value)
      .def("result", &ConstantOp::getResult);

  class_<CallIndirectOp>(m, "CallIndirectOp", cls)
      .def(init(&callIndirectCtor), "callee"_a, "operands"_a = ValueList{},
           "loc"_a = getUnknownLoc())
      .def_static("getName",
                  []() { return CallIndirectOp::getOperationName().str(); })
      .def("callee", &CallIndirectOp::callee)
      .def("operands", &CallIndirectOp::operands)
      .def("results",
           [](CallIndirectOp op) -> ValueRange { return op.results(); });
}

} // end namespace py
} // end namespace mlir
