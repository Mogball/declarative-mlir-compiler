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

LLVM::GlobalOp globalOpCtor(Type ty, bool isConstant, LLVM::Linkage linkage,
                            std::string name, Attribute value, Location loc) {
  OpBuilder b{getMLIRContext()};
  return b.create<LLVM::GlobalOp>(loc, ty.cast<LLVM::LLVMType>(), isConstant,
                                  static_cast<LLVM::Linkage>(linkage), name,
                                  value);
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
           "loc"_a = getUnknownLoc())
      .def_static("getName", []() { return ReturnOp::getOperationName().str(); });

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
      .def_static("getName", []() { return CallOp::getOperationName().str(); })
      .def("results",
           [](CallOp op) -> ValueRange { return op.getResults(); });

  class_<AddIOp>(m, "AddIOp", cls)
      .def(init(&addICtor), "lhs"_a, "rhs"_a, "ty"_a, "loc"_a = getUnknownLoc())
      .def("lhs", &AddIOp::lhs)
      .def("rhs", &AddIOp::rhs)
      .def("result", &AddIOp::getResult);

  class_<AddFOp>(m, "AddFOp", cls)
      .def(init([](Type ty, Value lhs, Value rhs, Location loc) {
        OpBuilder b{getMLIRContext()};
        return b.create<AddFOp>(loc, ty, lhs, rhs);
      }), "ty"_a, "lhs"_a, "rhs"_a, "loc"_a)
      .def("lhs", &AddFOp::lhs)
      .def("rhs", &AddFOp::rhs)
      .def("result", &AddFOp::getResult);

  class_<MulFOp>(m, "MulFOp", cls)
      .def(init([](Type ty, Value lhs, Value rhs, Location loc) {
        OpBuilder b{getMLIRContext()};
        return b.create<MulFOp>(loc, ty, lhs, rhs);
      }), "ty"_a, "lhs"_a, "rhs"_a, "loc"_a)
      .def("lhs", &MulFOp::lhs)
      .def("rhs", &MulFOp::rhs)
      .def("result", &MulFOp::getResult);

  class_<MulIOp>(m, "MulIOp", cls)
      .def(init([](Type ty, Value lhs, Value rhs, Location loc) {
        OpBuilder b{getMLIRContext()};
        return b.create<MulIOp>(loc, ty, lhs, rhs);
      }), "ty"_a, "lhs"_a, "rhs"_a, "loc"_a)
      .def("lhs", &MulIOp::lhs)
      .def("rhs", &MulIOp::rhs)
      .def("result", &MulIOp::getResult);

  class_<IndexCastOp>(m, "IndexCastOp", cls)
      .def(init([](Value source, Type type, Location loc) {
        OpBuilder b{getMLIRContext()};
        return b.create<IndexCastOp>(loc, source, type);
      }), "source"_a, "type"_a, "loc"_a = getUnknownLoc())
      .def("result", &IndexCastOp::getResult);

  class_<BranchOp>(m, "BranchOp", cls)
      .def(init([](Block *dest, ValueListRef destOperands, Location loc) {
        OpBuilder b{getMLIRContext()};
        return b.create<BranchOp>(loc, dest, destOperands);
      }), "dest"_a, "destOperands"_a = ValueList{}, "loc"_a = getUnknownLoc())
      .def("dest", &BranchOp::dest);

  class_<CondBranchOp>(m, "CondBranchOp", cls)
      .def(init([](Value cond, Block *trueDest, Block *falseDest,
                   ValueListRef trueOperands, ValueListRef falseOperands,
                   Location loc) {
        OpBuilder b{getMLIRContext()};
        return b.create<CondBranchOp>(loc, cond, trueDest, trueOperands,
                                      falseDest, falseOperands);
      }), "cond"_a, "trueDest"_a, "falseDest"_a,
           "trueOperands"_a = ValueList{}, "falseOperands"_a = ValueList{},
           "loc"_a = getUnknownLoc())
      .def("trueDest", &CondBranchOp::trueDest)
      .def("falseDest", &CondBranchOp::falseDest);

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
                  []() { return LLVM::LLVMFuncOp::getOperationName().str(); })
      .def_property_readonly("external", [](LLVM::LLVMFuncOp op) {
        return op.isExternal();
      })
      .def_property_readonly("name", [](LLVM::LLVMFuncOp op) {
        return op.getName().str();
      });

  class_<LLVM::UndefOp>(m, "LLVMUndefOp", cls)
      .def(init([](Type ty, Location loc) {
        OpBuilder b{getMLIRContext()};
        return b.create<LLVM::UndefOp>(loc, ty);
      }), "ty"_a, "loc"_a)
      .def("res", &LLVM::UndefOp::res);

  class_<LLVM::GlobalOp>(m, "LLVMGlobalOp", cls)
      .def(init([](Operation *op) { return cast<LLVM::GlobalOp>(op); }))
      .def(init(&globalOpCtor), "ty"_a, "isConstant"_a, "linkage"_a, "name"_a,
           "value"_a, "loc"_a = getUnknownLoc())
      .def("value", [](LLVM::GlobalOp op) { return op.valueAttr(); })
      .def_static("getName",
                  []() { return LLVM::GlobalOp::getOperationName().str(); });

  class_<LLVM::AllocaOp>(m, "LLVMAllocaOp", cls)
      .def(init([](Type res, Value arrSz, IntegerAttr align,
                   Location loc) {
        OpBuilder b{getMLIRContext()};
        return b.create<LLVM::AllocaOp>(loc, res, arrSz, align);
      }), "res"_a, "arrSz"_a, "align"_a, "loc"_a)
      .def("res", &LLVM::AllocaOp::res);

  class_<LLVM::GEPOp>(m, "LLVMGEPOp", cls)
      .def(init([](Type res, Value base, ValueListRef indices,
                   Location loc) {
        OpBuilder b{getMLIRContext()};
        return b.create<LLVM::GEPOp>(loc, res, base, indices);
      }), "res"_a, "base"_a, "indices"_a, "loc"_a)
      .def("res", &LLVM::GEPOp::res)
      .def_static("getName",
                  []() { return LLVM::GEPOp::getOperationName().str(); });

  class_<LLVM::StoreOp>(m, "LLVMStoreOp", cls)
      .def(init([](Value value, Value addr, Location loc) {
        OpBuilder b{getMLIRContext()};
        return b.create<LLVM::StoreOp>(loc, value, addr);
      }), "value"_a, "addr"_a, "loc"_a);

  class_<LLVM::LoadOp>(m, "LLVMLoadOp", cls)
      .def(init([](Type res, Value addr, Location loc) {
        OpBuilder b{getMLIRContext()};
        return b.create<LLVM::LoadOp>(loc, res, addr);
      }), "res"_a, "addr"_a, "loc"_a)
      .def("res", &LLVM::LoadOp::res);

  class_<LLVM::BitcastOp>(m, "LLVMBitcastOp", cls)
      .def(init([](Type res, Value arg, Location loc) {
        OpBuilder b{getMLIRContext()};
        return b.create<LLVM::BitcastOp>(loc, res, arg);
      }), "res"_a, "arg"_a, "loc"_a)
      .def("res", &LLVM::BitcastOp::res);

  class_<LLVM::AddressOfOp>(m, "LLVMAddressOfOp", cls)
      .def(init([](LLVM::GlobalOp global, Location loc) {
        OpBuilder b{getMLIRContext()};
        return b.create<LLVM::AddressOfOp>(loc, global);
      }), "value"_a, "loc"_a)
      .def("res", &LLVM::AddressOfOp::res)
      .def_static("getName",
                  []() { return LLVM::AddressOfOp::getOperationName().str(); });

  class_<LLVM::ConstantOp>(m, "LLVMConstantOp", cls)
      .def(init([](Type res, Attribute value, Location loc) {
        OpBuilder b{getMLIRContext()};
        return b.create<LLVM::ConstantOp>(loc, res, value);
      }), "res"_a, "value"_a, "loc"_a)
      .def("res", &LLVM::ConstantOp::res)
      .def_static("getName",
                  []() { return LLVM::ConstantOp::getOperationName().str(); });

  class_<LLVM::ExtractValueOp>(m, "LLVMExtractValueOp", cls)
      .def(init([](Type res, Value container, ArrayAttr pos,
                   Location loc) {
        OpBuilder b{getMLIRContext()};
        return b.create<LLVM::ExtractValueOp>(loc, res, container, pos);
      }), "res"_a, "container"_a, "pos"_a, "loc"_a)
      .def("res", &LLVM::ExtractValueOp::res);

  class_<LLVM::ZExtOp>(m, "LLVMZExtOp", cls)
      .def(init([](Type res, Value value, Location loc) {
        OpBuilder b{getMLIRContext()};
        return b.create<LLVM::ZExtOp>(loc, res, value);
      }), "res"_a, "value"_a, "loc"_a)
      .def("res", &LLVM::ZExtOp::res);

  class_<LLVM::TruncOp>(m, "LLVMTruncOp", cls)
      .def(init([](Type res, Value value, Location loc) {
        OpBuilder b{getMLIRContext()};
        return b.create<LLVM::TruncOp>(loc, res, value);
      }), "res"_a, "value"_a, "loc"_a)
      .def("res", &LLVM::TruncOp::res);

  class_<LLVM::InsertValueOp>(m, "LLVMInsertValueOp", cls)
      .def(init([](Type res, Value container, Value value,
                   ArrayAttr pos, Location loc) {
        OpBuilder b{getMLIRContext()};
        return b.create<LLVM::InsertValueOp>(loc, res, container, value, pos);
      }), "res"_a, "container"_a, "value"_a, "pos"_a, "loc"_a)
      .def("res", &LLVM::InsertValueOp::res);

  class_<LLVM::Linkage>(m, "LLVMLinkage")
      .def_static("External", []() { return LLVM::Linkage::External; })
      .def_static("Internal", []() { return LLVM::Linkage::Internal; });

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
