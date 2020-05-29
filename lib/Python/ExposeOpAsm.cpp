#include "Utility.h"
#include "Identifier.h"

#include <mlir/IR/OpImplementation.h>
#include <pybind11/pybind11.h>

using namespace pybind11;
using namespace mlir;

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
}

} // end namespace py
} // end namespace mlir
