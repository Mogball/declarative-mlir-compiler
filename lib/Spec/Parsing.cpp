#include "dmc/Spec/Parsing.h"
#include "dmc/Spec/Support.h"
#include "dmc/Spec/SpecRegionSwitch.h"
#include "dmc/Spec/SpecSuccessorSwitch.h"
#include "dmc/Spec/OpType.h"
#include "dmc/Spec/NamedConstraints.h"

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
      .Default(SpecRegion::LAST_SPEC_REGION);

  if (kind == SpecRegion::LAST_SPEC_REGION)
    return emitError(loc, "unknown region constraint: '") << name << '\'';
  ParseAction<Attribute, OpAsmParser> action{parser};
  opRegion = SpecRegion::kindSwitch(action, kind);
  return success(static_cast<bool>(opRegion));
}

void printOpRegion(llvm::raw_ostream &os, Attribute opRegion) {
  PrintAction<llvm::raw_ostream> action{os};
  SpecRegion::kindSwitch(action, opRegion);
}

ParseResult parseOptionalRegionList(OpAsmParser &parser, OpRegion &opRegion) {
  SmallVector<NamedConstraint, 1> regions;
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  if (!parser.parseOptionalLParen()) {
    do {
      NamedConstraint region;
      if (parser.parseKeyword(&region.name) || parser.parseColon() ||
          parseOpRegion(parser, region.attr))
        return failure();
      regions.push_back(std::move(region));
    } while (!parser.parseOptionalComma());
    if (parser.parseRParen())
      return failure();
  }
  opRegion = OpRegion::getChecked(loc, regions);
  return success(static_cast<bool>(opRegion));
}

template <typename PrinterT>
void printOptionalRegionList(PrinterT &printer, OpRegion opRegion) {
  if (opRegion.getNumRegions()) {
    printer << " (";
    llvm::interleaveComma(opRegion.getRegions(), printer,
        [&](const NamedConstraint &region) {
          printer << region.name << " : ";
          printOpRegion(printer.getStream(), region.attr);
        });
    printer << ')';
  }
}

template void
printOptionalRegionList(OpAsmPrinter &printer, OpRegion opRegion);
template void
printOptionalRegionList(DialectAsmPrinter &printer, OpRegion opRegion);

/// Successor parsing.
ParseResult parseOpSuccessor(OpAsmParser &parser, Attribute &opSucc) {
  StringRef name;
  if (parser.parseKeyword(&name))
    return failure();
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  auto kind = StringSwitch<unsigned>(name)
      .Case(AnySuccessor::getName(), SpecSuccessor::Any)
      .Case(VariadicSuccessor::getName(), SpecSuccessor::Variadic)
      .Default(SpecSuccessor::LAST_SPEC_SUCCESSOR);

  if (kind == SpecSuccessor::LAST_SPEC_SUCCESSOR)
    return emitError(loc, "unknown successor constraint: '") << name << '\'';
  ParseAction<Attribute, OpAsmParser> action{parser};
  opSucc = SpecSuccessor::kindSwitch(action, kind);
  return success(static_cast<bool>(opSucc));
}

void printOpSuccessor(llvm::raw_ostream &os, Attribute opSucc) {
  PrintAction<llvm::raw_ostream> action{os};
  SpecSuccessor::kindSwitch(action, opSucc);
}

ParseResult parseOptionalSuccessorList(OpAsmParser &parser,
                                       OpSuccessor &opSucc) {
  SmallVector<NamedConstraint, 2> succs;
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  if (!parser.parseOptionalLSquare()) {
    do {
      NamedConstraint succ;
      if (parser.parseKeyword(&succ.name) || parser.parseColon() ||
          parseOpSuccessor(parser, succ.attr))
        return failure();
      succs.push_back(std::move(succ));
    } while (!parser.parseOptionalComma());
    if (parser.parseRSquare())
      return failure();
  }
  opSucc = OpSuccessor::getChecked(loc, succs);
  return success(static_cast<bool>(opSucc));
}

template <typename PrinterT>
void printOptionalSuccessorList(PrinterT &printer, OpSuccessor opSucc) {
  if (opSucc.getNumSuccessors()) {
    printer << " [";
    llvm::interleaveComma(opSucc.getSuccessors(), printer,
        [&](const NamedConstraint &succ) {
          printer << succ.name << " : ";
          printOpSuccessor(printer.getStream(), succ.attr);
        });
    printer << ']';
  }
}

template void
printOptionalSuccessorList(OpAsmPrinter &printer, OpSuccessor opSucc);
template void
printOptionalSuccessorList(DialectAsmPrinter &printer, OpSuccessor opSucc);

/// OpType parsing.
template <typename NamedTypeList>
ParseResult parseValueList(OpAsmParser &parser, NamedTypeList &tys) {
  NamedType ty;
  if (parser.parseLParen())
    return failure();
  // Parse the first value, if present
  if (!parser.parseOptionalKeyword(&ty.name)) {
    if (parser.parseColon() || parser.parseType(ty.type))
      return failure();
    tys.push_back(std::move(ty));
    // Parse list values
    while (!parser.parseOptionalComma()) {
      if (parser.parseKeyword(&ty.name) || parser.parseColon() ||
          parser.parseType(ty.type))
        return failure();
      tys.push_back(std::move(ty));
    }
  }
  return parser.parseRParen();
}

ParseResult parseOpType(OpAsmParser &parser, OpType &opType) {
  // op-type ::= `(` value-list `)` `->` `(` value-list `)`
  // value-list ::= (value (`,` value)*)?
  // value ::= identifier `:` type
  SmallVector<NamedType, 4> operands, results;
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  if (parseValueList(parser, operands) || parser.parseArrow() ||
      parseValueList(parser, results))
    return failure();
  opType = OpType::getChecked(loc, operands, results);
  return success();
}

template <typename PrinterT, typename NamedTypeList>
void printValueList(PrinterT &printer, NamedTypeList types) {
  printer << '(';
  llvm::interleaveComma(types, printer, [&](const NamedType &ty) {
    printer << ty.name << " : " << ty.type;
  });
  printer << ')';
}

template <typename PrinterT>
void printOpType(PrinterT &printer, OpType opType) {
  printValueList(printer, opType.getOperands());
  printer << " -> ";
  printValueList(printer, opType.getResults());
}

template void printOpType(OpAsmPrinter &printer, OpType opType);
template void printOpType(DialectAsmPrinter &printer, OpType opType);

} // end namespace impl
} // end namespace dmc
