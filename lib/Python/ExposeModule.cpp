#include "Expose.h"
#include "Context.h"
#include "Location.h"
#include "Module.h"
#include "Utility.h"
#include "Identifier.h"
#include "OwningModuleRef.h"

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/SCF.h>

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

CallIndirectOp callIndirectCtor(Value callee, ValueListRef operands,
                                Location loc) {
  OpBuilder b{getMLIRContext()};
  return b.create<CallIndirectOp>(loc, callee, operands);
}

AddIOp addICtor(Value lhs, Value rhs, Type ty, Location loc) {
  OpBuilder b{getMLIRContext()};
  return b.create<AddIOp>(loc, ty, lhs, rhs);
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
      .def("lookup", [](ModuleOp module, std::string name) {
        return module.lookupSymbol(name);
      }, return_value_policy::reference)
      .def("append", nullcheck(&ModuleOp::push_back), "op"_a)
      .def("insertBefore",
           nullcheck(overload<void(ModuleOp::*)(Operation *, Operation *)>(
               &ModuleOp::insert)),
           "insertPt"_a, "op"_a)
      .def_static("getName",
                  []() { return ModuleOp::getOperationName().str(); });

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
      .def(init([](Attribute value, Location loc) {
        OpBuilder b{getMLIRContext()};
        return b.create<ConstantOp>(loc, value);
      }), "value"_a, "loc"_a = getUnknownLoc())
      .def(init([](Attribute value, Type ty, Location loc) {
        OpBuilder b{getMLIRContext()};
        return b.create<ConstantOp>(loc, ty, value);
      }), "value"_a, "ty"_a, "loc"_a = getUnknownLoc())
      .def(init([](Operation *op) { return cast<ConstantOp>(op); }))
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

  class_<CallOp>(m, "CallOp", cls)
      .def(init([](FuncOp callee, ValueListRef operands, Location loc) {
        OpBuilder b{getMLIRContext()};
        return b.create<CallOp>(loc, callee, operands);
      }), "callee"_a, "operands"_a, "loc"_a = getUnknownLoc())
      .def_static("getName", []() { return CallOp::getOperationName().str(); });

  class_<AddIOp>(m, "AddIOp", cls)
      .def(init(&addICtor), "lhs"_a, "rhs"_a, "ty"_a, "loc"_a = getUnknownLoc())
      .def("lhs", &AddIOp::lhs)
      .def("rhs", &AddIOp::rhs)
      .def("result", &AddIOp::getResult);

  class_<IndexCastOp>(m, "IndexCastOp", cls)
      .def(init([](Value source, Type type, Location loc) {
        OpBuilder b{getMLIRContext()};
        return b.create<IndexCastOp>(loc, source, type);
      }), "source"_a, "type"_a, "loc"_a = getUnknownLoc())
      .def("result", &IndexCastOp::getResult);

  class_<LLVM::CallOp>(m, "LLVMCallOp", cls)
      .def(init([](LLVM::LLVMFuncOp func, ValueListRef operands, Location loc) {
        OpBuilder b{getMLIRContext()};
        return b.create<LLVM::CallOp>(loc, func, operands);
      }), "func"_a, "operands"_a, "loc"_a = getUnknownLoc())
      .def_static("getName",
                  []() { return LLVM::CallOp::getOperationName().str(); });
  class_<LLVM::LLVMFuncOp>(m, "LLVMFuncOp", cls)
      .def(init([](Operation *op) { return cast<LLVM::LLVMFuncOp>(op); }))
      .def_static("getName",
                  []() { return LLVM::LLVMFuncOp::getOperationName().str(); });

  class_<LLVM::GlobalOp>(m, "LLVMGlobalOp", cls)
      .def(init([](Operation *op) { return cast<LLVM::GlobalOp>(op); }))
      .def_static("getName",
                  []() { return LLVM::GlobalOp::getOperationName().str(); });

  class_<scf::ForOp>(m, "ForOp", cls)
      .def(init([](Value lowerBound, Value upperBound, Value step,
                   Location loc) {
        OpBuilder b{getMLIRContext()};
        return b.create<scf::ForOp>(loc, lowerBound, upperBound, step);
      }), "lowerBound"_a, "upperBound"_a, "step"_a, "loc"_a = getUnknownLoc())
      .def("getInductionVar", &scf::ForOp::getInductionVar)
      .def("region", &scf::ForOp::region, return_value_policy::reference);

  class_<scf::YieldOp>(m, "YieldOp", cls)
      .def(init([](Location loc) {
        OpBuilder b{getMLIRContext()};
        return b.create<scf::YieldOp>(loc);
      }), "loc"_a = getUnknownLoc());
}

} // end namespace py
} // end namespace mlir
