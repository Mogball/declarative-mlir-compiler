#include "Utility.h"
#include "Identifier.h"
#include "dmc/Python/OpAsm.h"
#include "dmc/Traits/SpecTraits.h"

#include <mlir/IR/OpImplementation.h>
#include <pybind11/pybind11.h>

using namespace pybind11;
using namespace mlir;

using dmc::py::OperationWrap;

namespace mlir {
namespace py {

auto transformAttrStorage(AttrDictRef attrs, StringListRef elidedAttrs) {
  /// Convert unordered_map<string, Attribute> -> vector<{Identifier, Attribute}>
  /// and vector<string> -> vector<StringRef>
  NamedAttrList namedAttrs;
  for (auto &[name, attr] : attrs)
    namedAttrs.push_back({getIdentifierChecked(name), attr});
  std::vector<StringRef> refs;
  refs.reserve(std::size(elidedAttrs));
  for (auto &elidedAttr : elidedAttrs)
    refs.emplace_back(elidedAttr);
  return std::make_pair(std::move(namedAttrs), std::move(refs));
}

template <typename E> bool is_list_of(const list &vals) {
  return llvm::all_of(vals, [](auto &val) { return isinstance<E>(val); });
}

template <typename E> auto map_list(const list &vals) {
  return llvm::map_range(vals, [](auto &val) { return val.template cast<E>(); });
}

void genericPrint(OpAsmPrinter &printer, const object &obj) {
  if (isinstance<Value>(obj)) {
    printer.printOperand(obj.cast<Value>());
  } else if (isinstance<list>(obj)) {
    auto vals = obj.cast<list>();
    if (is_list_of<Value>(vals)) {
      printer.printOperands(map_list<Value>(vals));
    } else if (is_list_of<Type>(vals)) {
      llvm::interleaveComma(map_list<Type>(vals), printer);
    } else {
      llvm::interleaveComma(vals, printer,
                            [&](auto &val) { genericPrint(printer, obj); });
    }
  } else if (isinstance<iterator>(obj)) {
    /// An iterator can only be exhausted once and the internet state cannot be
    /// copied, so empty its contents into a list.
    auto it = obj.cast<iterator>();
    list itList;
    while (it != iterator::sentinel())
      itList.append(*it++);
    genericPrint(printer, itList);
  } else if (isinstance<Type>(obj)) {
    printer.printType(obj.cast<Type>());
  } else if (isinstance<Attribute>(obj)) {
    printer.printAttribute(obj.cast<Attribute>());
  } else if (isinstance<bool>(obj)) {
    printer << obj.cast<bool>();
  } else if (isinstance<Block>(obj)) {
    printer.printSuccessor(&obj.cast<Block &>());
  } else {
    std::string repr = str(obj);
    printer << repr;
  }
}

static Value getOperand(OperationWrap &op, unsigned idx) {
  ValueRange (*getGroup)(Operation *, unsigned){};
  if (op.getSpec()->getTrait<dmc::SizedOperandSegments>()) {
    getGroup = &dmc::SizedOperandSegments::getOperandGroup;
  } else if (op.getSpec()->getTrait<dmc::SameVariadicOperandSizes>()) {
    getGroup = &dmc::SameVariadicOperandSizes::getOperandGroup;
  } else {
    if (idx >= op.getOp()->getNumOperands()) {
      throw std::invalid_argument{
          "Op does not have the expected number of operands"};
    }
    return op.getOp()->getOperand(idx); // no variadic operands
  }
  auto group = getGroup(op.getOp(), idx);
  if (llvm::size(group) != 1)
    throw std::invalid_argument{"Expected a singular operand group"};
  return *group.begin();
}

static Value getResult(OperationWrap &op, unsigned idx) {
  ValueRange (*getGroup)(Operation *, unsigned){};
  if (op.getSpec()->getTrait<dmc::SizedResultSegments>()) {
    getGroup = &dmc::SizedResultSegments::getResultGroup;
  } else if (op.getSpec()->getTrait<dmc::SameVariadicResultSizes>()) {
    getGroup = &dmc::SameVariadicResultSizes::getResultGroup;
  } else {
    if (idx >= op.getOp()->getNumResults()) {
      throw std::invalid_argument{
          "Op does not have the expected number of results"};
    }
    return op.getOp()->getResult(idx);
  }
  auto group = getGroup(op.getOp(), idx);
  if (llvm::size(group) != 1)
    throw std::invalid_argument{"Expected a singular result group"};
  return *group.begin();
}

void exposeOpAsm(module &m) {
  class_<OpAsmPrinter, std::unique_ptr<OpAsmPrinter, nodelete>>(m, "OpAsmPrinter")
      .def("printOperand", overload<void(OpAsmPrinter::*)(Value)>(
          &OpAsmPrinter::printOperand))
      .def("printOperands", [](OpAsmPrinter &printer, ValueListRef values) {
        printer.printOperands(values);
      })
      .def("printType", &OpAsmPrinter::printType)
      .def("printAttribute", &OpAsmPrinter::printAttribute)
      .def("printAttributeWithoutType", &OpAsmPrinter::printAttributeWithoutType)
      .def("printSuccessor", &OpAsmPrinter::printSuccessor)
      .def("printSuccessorAndUseList", [](OpAsmPrinter &printer, Block *successor,
                                          ValueListRef succOperands) {
        printer.printSuccessorAndUseList(successor, succOperands);
      })
      .def("printOptionalAttrDict", [](OpAsmPrinter &printer, AttrDictRef attrs,
                                       StringListRef elidedAttrs) {
        auto [namedAttrs, refs] = transformAttrStorage(attrs, elidedAttrs);
        printer.printOptionalAttrDict(namedAttrs, refs);
      }, "attrs"_a, "elidedAttrs"_a = StringList{})
      .def("printOptionalAttrDictWithKeyword",
           [](OpAsmPrinter &printer, AttrDictRef attrs, StringListRef elidedAttrs) {
        auto [namedAttrs, refs] = transformAttrStorage(attrs, elidedAttrs);
        printer.printOptionalAttrDictWithKeyword(namedAttrs, refs);
      }, "attrs"_a, "elidedAttrs"_a = StringList{})
      .def("printGenericOp", &OpAsmPrinter::printGenericOp)
      .def("printRegion", &OpAsmPrinter::printRegion, "region"_a,
           "printEntryBlockArgs"_a = true, "printBlockTerminators"_a = true)
      .def("shadowRegionArgs", [](OpAsmPrinter &printer, Region &region,
                                  ValueListRef namesToUse) {
        printer.shadowRegionArgs(region, namesToUse);
      })
      .def("printAffineMapOfSSAIds", [](OpAsmPrinter &printer, AffineMapAttr mapAttr,
                                        ValueListRef operands) {
        printer.printAffineMapOfSSAIds(mapAttr, operands);
      })
      .def("printOptionalArrowTypeList",
           &OpAsmPrinter::printOptionalArrowTypeList<TypeListRef>)
      .def("printArrowTypeList", &OpAsmPrinter::printArrowTypeList<TypeListRef>)
      .def("printFunctionalType",
           overload<void(OpAsmPrinter::*)(Operation *)>(&OpAsmPrinter::printFunctionalType))
      .def("printFunctionalType",
           &OpAsmPrinter::printFunctionalType<TypeListRef, TypeListRef>)
      .def("print", [](OpAsmPrinter &printer, pybind11::args args) {
        for (auto &val : args)
          genericPrint(printer, reinterpret_borrow<object>(val));
      });

  class_<OperationWrap>(m, "OperationWrap")
      .def("getAttrs", [](OperationWrap &op) {
        return pybind11::cast(op.getOp()).attr("getAttrs")().cast<dict>();
      })
      .def("getAttr", [](OperationWrap &op, std::string name) {
        return op.getOp()->getAttr(name);
      })
      .def("getOperandTypes", [](OperationWrap &op) {
        list types;
        for (auto type : op.getOp()->getOperandTypes())
          types.append(pybind11::cast(type));
        return types;
      })
      .def("getResultTypes", [](OperationWrap &op) {
        list types;
        for (auto type : op.getOp()->getResultTypes())
          types.append(pybind11::cast(type));
        return types;
      })
      .def("getOperand", [](OperationWrap &op, std::string name) {
        auto opTy = op.getType()->getOpType();
        unsigned idx = 0;
        for (auto &operand : opTy.getOperands()) {
          if (operand.name == name)
            return getOperand(op, idx);
          ++idx;
        }
        throw std::invalid_argument{op.getSpec()->getName() +
                                    " does not have an operand named '" +
                                    name + "'"};
      })
      .def("getResult", [](OperationWrap &op, std::string name) {
        auto opTy = op.getType()->getOpType();
        unsigned idx = 0;
        for (auto &result : opTy.getResults()) {
          if (result.name == name)
            return getResult(op, idx);
          ++idx;
        }
        throw std::invalid_argument{op.getSpec()->getName() +
                                    " does not have a result named '" +
                                    name + "'"};
      });
}

} // end namespace py
} // end namespace mlir
