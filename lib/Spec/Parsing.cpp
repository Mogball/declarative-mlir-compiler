#include "dmc/Spec/Parsing.h"

#include <mlir/IR/Builders.h>

using namespace mlir;

namespace dmc {
namespace impl {

/// Parse a single attribute. OpAsmPrinter provides only an API to parse
/// a NamedAttribute into a list.
ParseResult parseSingleAttribute(OpAsmParser &parser, Attribute &attr) {
  NamedAttrList attrList;
  return parser.parseAttribute(attr, "single", attrList);
}

/// Parse an optional parameter list for any parser.
namespace detail {
template <typename ParserT, typename ParseAttrFn, typename ModifierFn>
ParseResult parseOptionalParameterList(
    ParserT &parser, mlir::ArrayAttr &attr, ParseAttrFn parseAttr,
    ModifierFn modifier) {
  SmallVector<Attribute, 2> params;
  if (!parser.parseOptionalLess()) {
    do {
      Attribute param;
      if (parseAttr(parser, param))
        return failure();
      params.push_back(modifier(param));
    } while (!parser.parseOptionalComma());
    if (parser.parseGreater())
      return failure();
  }
  attr = parser.getBuilder().getArrayAttr(params);
  return success();
}
} // end namespace detail

/// Parse an optional parameter list with an on-the-fly modifier. Callers can
/// use the modifier to intercept certain parameters without re-traversing the
/// resultant attribute array.
ParseResult parseOptionalParameterList(
    OpAsmParser &parser, mlir::ArrayAttr &attr,
    Attribute (modifier)(Attribute)) {
  return detail::parseOptionalParameterList(parser, attr, parseSingleAttribute,
                                            modifier);
}

ParseResult parseOptionalParameterList(
    DialectAsmParser &parser, mlir::ArrayAttr &attr,
    Attribute (modifier)(Attribute)) {
  auto parseAttr = [](DialectAsmParser &parser, Attribute &attr) {
    return parser.parseAttribute(attr);
  };
  return detail::parseOptionalParameterList(parser, attr, parseAttr, modifier);
}

/// Parse an optional parameter list. If not found, returns success() and an
/// empty array attribute.
///
/// parameter-list ::= (`<` attr (`,` attr)* `>`)?
ParseResult parseOptionalParameterList(OpAsmParser &parser,
                                       mlir::ArrayAttr &attr) {
  return parseOptionalParameterList(parser, attr,
                                    [](Attribute attr) { return attr; });
}

ParseResult parseOptionalParameterList(DialectAsmParser &parser,
                                       mlir::ArrayAttr &attr) {
  return parseOptionalParameterList(parser, attr,
                                    [](Attribute attr) { return attr; });
}

/// Print a parameter list for any printer.
namespace detail {
template <typename PrinterT, typename AttrRange>
void printOptionalParameterList(PrinterT &printer, AttrRange params) {
  if (!params.empty()) {
    auto it = std::begin(params);
    printer << '<' << *it++;
    for (auto e = std::end(params); it != e; ++it)
      printer << ',' << *it;
    printer << '>';
  }
}
} // end namespace detail

void printOptionalParameterList(OpAsmPrinter &printer,
                                ArrayRef<Attribute> params) {
  return detail::printOptionalParameterList(printer, params);
}

void printOptionalParameterList(DialectAsmPrinter &printer,
                        ArrayRef<Attribute> params) {
  return detail::printOptionalParameterList(printer, params);
}

/// Op trait parsing.
ParseResult parseOpTrait(OpAsmParser &parser, OpTraitAttr &traitAttr) {
  // op-trait ::= `@`trait-name parameter-list?
  mlir::StringAttr nameAttr;
  mlir::ArrayAttr paramAttr;
  NamedAttrList attrList; // dummy list
  if (parser.parseSymbolName(nameAttr, "name", attrList) ||
      parseOptionalParameterList(parser, paramAttr))
    return failure();
  /// TODO getChecked, but OpAsmParser cannot convert SMLoc to Location
  traitAttr = OpTraitAttr::get(
      parser.getBuilder().getSymbolRefAttr(nameAttr.getValue()), paramAttr);
  return success();
}

ParseResult parseOptionalOpTraitList(OpAsmParser &parser,
                                     OpTraitsAttr &traitArr) {
  // trait-list ::= `traits` `[` op-trait (`,` op-trait)* `]`
  if (parser.parseOptionalKeyword("traits"))
    return success();
  if (parser.parseLSquare())
    return failure();
  SmallVector<Attribute, 2> opTraits;
  do {
    OpTraitAttr traitAttr;
    if (parseOpTrait(parser, traitAttr))
      return failure();
    opTraits.push_back(traitAttr);
  } while (!parser.parseOptionalComma());
  if (parser.parseRSquare())
    return failure();
  /// TODO getChecked, but OpAsmParser cannot convert SMLoc to Location
  traitArr = OpTraitsAttr::get(parser.getBuilder().getArrayAttr(opTraits));
  return success();
}

void printOpTrait(OpAsmPrinter &printer, OpTraitAttr traitAttr) {
  printer.printSymbolName(traitAttr.getName());
  printOptionalParameterList(printer, traitAttr.getParameters());
}

void printOptionalOpTraitList(OpAsmPrinter &printer, OpTraitsAttr traitArr) {
  auto traits = traitArr.getValue();
  if (traits.empty())
    return;
  auto it = std::begin(traits);
  printer << "traits [";
  printOpTrait(printer, *it++);
  for (auto e = std::end(traits); it != e; ++it) {
    printer << ',';
    printOpTrait(printer, *it);
  }
  printer << ']';
}

} // end namespace impl
} // end namespace dmc
