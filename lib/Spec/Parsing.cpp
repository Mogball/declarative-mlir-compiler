#include "dmc/Spec/Parsing.h"
#include "dmc/Spec/Support.h"
#include "dmc/Spec/SpecRegion.h"
#include "dmc/Spec/SpecRegionSwitch.h"

#include <mlir/IR/Builders.h>
#include <llvm/ADT/StringSwitch.h>

using namespace mlir;
using namespace llvm;

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
    OpAsmParser &parser, mlir::ArrayAttr &attr, ParameterModifier modifier) {
  return detail::parseOptionalParameterList(parser, attr, parseSingleAttribute,
                                            modifier);
}

ParseResult parseOptionalParameterList(
    DialectAsmParser &parser, mlir::ArrayAttr &attr,
    ParameterModifier modifier) {
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
    printer << '<';
    llvm::interleaveComma(params, printer,
                          [&](Attribute attr) { printer << attr; });
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
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  mlir::StringAttr nameAttr;
  mlir::ArrayAttr paramAttr;
  NamedAttrList attrList; // dummy list
  if (parser.parseSymbolName(nameAttr, "name", attrList) ||
      parseOptionalParameterList(parser, paramAttr))
    return failure();
  traitAttr = OpTraitAttr::getChecked(loc, parser.getBuilder().getSymbolRefAttr(
      nameAttr.getValue()), paramAttr);
  return success();
}

ParseResult parseOptionalOpTraitList(OpAsmParser &parser,
                                     OpTraitsAttr &traitArr) {
  // trait-list ::= `traits` `[` op-trait (`,` op-trait)* `]`
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  SmallVector<Attribute, 2> opTraits;
  if (!parser.parseOptionalKeyword("traits")) {
    if (parser.parseLSquare())
      return failure();
    do {
      OpTraitAttr traitAttr;
      if (parseOpTrait(parser, traitAttr))
        return failure();
      opTraits.push_back(traitAttr);
    } while (!parser.parseOptionalComma());
    if (parser.parseRSquare())
      return failure();
  }
  traitArr = OpTraitsAttr::getChecked(
      loc, parser.getBuilder().getArrayAttr(opTraits));
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
  printer << " traits [";
  llvm::interleaveComma(traits, printer,
                        [&](auto trait) { printOpTrait(printer, trait); });
  printer << ']';
}

/// Region parsing.
ParseResult parseOpRegion(OpAsmParser &parser, Attribute &opRegion) {
  StringRef name;
  if (parser.parseKeyword(&name))
    return failure();
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  auto kind = StringSwitch<unsigned>(name)
      .Case(AnyRegion::getName(), SpecRegion::Any)
      .Case(SizedRegion::getName(), SpecRegion::Sized)
      .Case(IsolatedFromAboveRegion::getName(), SpecRegion::IsolatedFromAbove)
      .Case(VariadicRegion::getName(), SpecRegion::Variadic)
      .Default(SpecRegion::NUM_KINDS);

  if (kind == SpecRegion::NUM_KINDS)
    return emitError(loc, "unknown region constraint: '") << name << '\'';
  ParseAction<Attribute, OpAsmParser> action{parser};
  opRegion = SpecRegion::kindSwitch(action, kind);
  return success(static_cast<bool>(opRegion));
}

void printOpRegion(OpAsmPrinter &printer, Attribute opRegion) {
  printOpRegion(printer.getStream(), opRegion);
}

void printOpRegion(llvm::raw_ostream &os, Attribute opRegion) {
  PrintAction<llvm::raw_ostream> action{os};
  SpecRegion::kindSwitch(action, opRegion);
}

ParseResult parseOptionalRegionList(OpAsmParser &parser,
                                    mlir::ArrayAttr &regionsAttr) {
  SmallVector<Attribute, 1> opRegions;
  if (!parser.parseOptionalLParen()) {
    do {
      Attribute opRegion;
      if (parseOpRegion(parser, opRegion))
        return failure();
      opRegions.push_back(opRegion);
    } while (!parser.parseOptionalComma());
    if (parser.parseRParen())
      return failure();
  }
  regionsAttr = parser.getBuilder().getArrayAttr(opRegions);
  return success();
}

void printOptionalRegionList(OpAsmPrinter &printer,
                             mlir::ArrayAttr regionsAttr) {
  if (llvm::size(regionsAttr)) {
    printer << " (";
    llvm::interleaveComma(regionsAttr, printer, [&](Attribute region)
                          { printOpRegion(printer, region); });
    printer << ')';
  }
}

} // end namespace impl
} // end namespace dmc
