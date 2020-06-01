#include "dmc/Spec/SpecOps.h"
#include "dmc/Spec/SpecTypes.h"
#include "dmc/Spec/SpecAttrs.h"
#include "dmc/Spec/SpecSuccessor.h"
#include "dmc/Spec/Parsing.h"
#include "dmc/Traits/SpecTraits.h"
#include "dmc/Traits/Registry.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>

using namespace mlir;
using namespace mlir::dmc;

namespace dmc {

void addAttributesIfNotPresent(ArrayRef<NamedAttribute> attrs,
                               OperationState &result) {
  llvm::DenseSet<StringRef> present;
  present.reserve(llvm::size(result.attributes));
  for (auto &namedAttr : result.attributes)
    present.insert(namedAttr.first);
  for (auto &namedAttr : attrs)
    if (present.find(namedAttr.first) == std::end(present))
      result.attributes.push_back(namedAttr);
}

/// DialectOp.
void DialectOp::buildDefaultValuedAttrs(OpBuilder &builder,
                                        OperationState &result) {
  auto falseAttr = builder.getBoolAttr(false);
  addAttributesIfNotPresent({
      builder.getNamedAttr(getAllowUnknownOpsAttrName(), falseAttr),
      builder.getNamedAttr(getAllowUnknownTypesAttrName(), falseAttr)},
      result);
}

void DialectOp::build(OpBuilder &builder, OperationState &result,
                      StringRef name) {
  ensureTerminator(*result.addRegion(), builder, result.location);
  result.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
      builder.getStringAttr(name));
  buildDefaultValuedAttrs(builder, result);
}

/// dialect-op ::= `Dialect` `@`symbolName `attributes` attr-list region
ParseResult DialectOp::parse(OpAsmParser &parser, OperationState &result) {
  mlir::StringAttr nameAttr;
  auto *body = result.addRegion();
  if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
      parser.parseRegion(*body, llvm::None, llvm::None))
    return failure();
  OpBuilder builder{result.getContext()};
  buildDefaultValuedAttrs(builder, result);
  ensureTerminator(*body, parser.getBuilder(), result.location);
  return success();
}

void DialectOp::print(OpAsmPrinter &printer) {
  printer << getOperation()->getName() << ' ';
  printer.printSymbolName(getName());
  printer.printOptionalAttrDictWithKeyword(getAttrs(), {
      mlir::SymbolTable::getSymbolAttrName()});
  printer.printRegion(getBodyRegion(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/false);
}

LogicalResult DialectOp::verify() {
  auto &region = getBodyRegion();
  if (!llvm::hasSingleElement(region))
    return emitOpError("expected Dialect body to have one block");
  auto *body = getBody();
  if (body->getNumArguments() != 0)
    return emitOpError("expected Dialect body to have zero arguments");
  /// Verify attributes
  if (!getAttrOfType<mlir::BoolAttr>(getAllowUnknownOpsAttrName()))
    return emitOpError("expected BoolAttr named: ")
        << getAllowUnknownOpsAttrName();
  if (!getAttrOfType<mlir::BoolAttr>(getAllowUnknownTypesAttrName()))
    return emitOpError("expected BoolAttr named: ")
        << getAllowUnknownTypesAttrName();
  return success();
}

Region &DialectOp::getBodyRegion() { return getOperation()->getRegion(0); }
Block *DialectOp::getBody() { return &getBodyRegion().front(); }

bool DialectOp::allowsUnknownOps() {
  return getAttrOfType<mlir::BoolAttr>(getAllowUnknownOpsAttrName())
      .getValue();
}

bool DialectOp::allowsUnknownTypes() {
  return getAttrOfType<mlir::BoolAttr>(getAllowUnknownTypesAttrName())
      .getValue();
}

/// OperationOp
void OperationOp::setOpType(OpType opTy) {
  setAttr(getOpTypeAttrName(), mlir::TypeAttr::get(opTy));
}

void OperationOp::setOpAttrs(mlir::DictionaryAttr opAttrs) {
  setAttr(getOpAttrDictAttrName(), opAttrs);
}

void OperationOp::setOpRegions(mlir::ArrayAttr opRegions) {
  setAttr(getOpRegionsAttrName(), opRegions);
}

void OperationOp::setOpSuccessors(mlir::ArrayAttr opSuccs) {
  setAttr(getOpSuccsAttrName(), opSuccs);
}

void OperationOp::buildDefaultValuedAttrs(OpBuilder &builder,
                                          OperationState &result) {
  auto falseAttr = builder.getBoolAttr(false);
  addAttributesIfNotPresent({
      builder.getNamedAttr(getIsTerminatorAttrName(), falseAttr),
      builder.getNamedAttr(getIsCommutativeAttrName(), falseAttr),
      builder.getNamedAttr(getIsIsolatedFromAboveAttrName(), falseAttr),
      builder.getNamedAttr(getOpTraitsAttrName(), builder.getArrayAttr({}))},
      result);
}

void OperationOp::build(
    OpBuilder &builder, OperationState &result, StringRef name, OpType opType,
    ArrayRef<NamedAttribute> opAttrs,
    ArrayRef<Attribute> opRegions,
    ArrayRef<Attribute> opSuccs,
    ArrayRef<NamedAttribute> config) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute(getOpTypeAttrName(), mlir::TypeAttr::get(opType));
  result.addAttribute(getOpAttrDictAttrName(),
                      builder.getDictionaryAttr(opAttrs));
  result.addAttribute(getOpRegionsAttrName(),
                      builder.getArrayAttr(opRegions));
  result.addAttribute(getOpSuccsAttrName(),
                      builder.getArrayAttr(opSuccs));
  result.attributes.append(std::begin(config), std::end(config));
  result.addRegion();
  buildDefaultValuedAttrs(builder, result);
}

// op ::= `dmc.Op` `@`op-name func-type (attr-dict)? (region-list)? (succ-list)?
//        (`traits` trait-list)?
//        (`config` attr-dict)?
//
// func-type ::= type-list `->` type-list
// trait-list ::= `[` trait (`,` trait)* `]`
// trait ::= `@`trait-name param-list?
// region-list ::= `(` region-attr (`,` region-attr)* `)`
// succ-list ::= `[` succ-attr (`,` succ-attr)* `]`
ParseResult OperationOp::parse(OpAsmParser &parser, OperationState &result) {
  mlir::StringAttr nameAttr;
  OpType opType;
  NamedAttrList opAttrs;
  mlir::ArrayAttr regionAttr;
  mlir::ArrayAttr succAttr;
  OpTraitsAttr traitArr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      impl::parseOpType(parser, opType) ||
      parser.parseOptionalAttrDict(opAttrs) ||
      impl::parseOptionalRegionList(parser, regionAttr) ||
      impl::parseOptionalSuccessorList(parser, succAttr) ||
      impl::parseOptionalOpTraitList(parser, traitArr))
    return failure();
  result.addAttribute(getOpTypeAttrName(), mlir::TypeAttr::get(opType));
  result.addAttribute(getOpAttrDictAttrName(),
                      mlir::DictionaryAttr::get(opAttrs, result.getContext()));
  result.addAttribute(getOpRegionsAttrName(), regionAttr);
  result.addAttribute(getOpSuccsAttrName(), succAttr);
  result.addAttribute(getOpTraitsAttrName(), traitArr);
  if (!parser.parseOptionalKeyword("config") &&
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.addRegion();
  OpBuilder builder{result.getContext()};
  buildDefaultValuedAttrs(builder, result);
  return success();
}

void OperationOp::print(OpAsmPrinter &printer) {
  printer << getOperation()->getName() << ' ';
  printer.printSymbolName(getName());
  impl::printOpType(printer, getOpType());
  printer << ' ';
  printer.printAttribute(getOpAttrs());
  impl::printOptionalRegionList(printer, getOpRegions());
  impl::printOptionalOpTraitList(printer, getOpTraits());
  impl::printOptionalSuccessorList(printer, getOpSuccessors());
  printer << " config";
  printer.printOptionalAttrDict(getAttrs(), {
      SymbolTable::getSymbolAttrName(), getOpTypeAttrName(),
      getOpAttrDictAttrName(), getOpRegionsAttrName(),
      getOpSuccsAttrName(), getOpTraitsAttrName()});
}

OpType OperationOp::getOpType() {
  return getAttrOfType<mlir::TypeAttr>(getOpTypeAttrName()).getValue()
      .cast<OpType>();
}

mlir::DictionaryAttr OperationOp::getOpAttrs() {
  return getAttrOfType<mlir::DictionaryAttr>(getOpAttrDictAttrName());
}

mlir::ArrayAttr OperationOp::getOpRegions() {
  return getAttrOfType<mlir::ArrayAttr>(getOpRegionsAttrName());
}

mlir::ArrayAttr OperationOp::getOpSuccessors() {
  return getAttrOfType<mlir::ArrayAttr>(getOpSuccsAttrName());
}

OpTraitsAttr OperationOp::getOpTraits() {
  return getAttrOfType<OpTraitsAttr>(getOpTraitsAttrName());
}

bool OperationOp::isTerminator() {
  return getAttrOfType<mlir::BoolAttr>(getIsTerminatorAttrName()).getValue();
}

bool OperationOp::isCommutative() {
  return getAttrOfType<mlir::BoolAttr>(getIsCommutativeAttrName()).getValue();
}

bool OperationOp::isIsolatedFromAbove() {
  return getAttrOfType<mlir::BoolAttr>(getIsIsolatedFromAboveAttrName())
      .getValue();
}

unsigned OperationOp::getNumOperands() { return getOpType().getNumOperands(); }
unsigned OperationOp::getNumResults() { return getOpType().getNumResults(); }

/// Trait array manipulation helpers.
namespace impl {
template <typename TraitT> bool hasTrait(OpTraitsAttr opTraits) {
  return llvm::count_if(opTraits.getValue(), [](OpTraitAttr sym)
                        { return sym.getName() == TraitT::getName(); });
}

/// Check that all constraints satisfy `is` and only the last is variadic.
template <typename VariadicT, typename ListT, typename CheckFcn>
LogicalResult verifyConstraintList(Operation *op, ListT vars, CheckFcn &&is,
                                   const char *name) {
  /// Ensure that op regions are valid region constraints.
  unsigned idx{}, numVar{};
  for (auto var : vars) {
    if (!is(var))
      return op->emitOpError("expected a valid ") << name << " constraint for "
          << name << " #" << idx;
    if (var.template isa<VariadicT>())
      ++numVar;
    ++idx;
  }
  /// Check at most one variadic region, which must be the last constraint.
  if (numVar) {
    if (numVar > 1)
      return op->emitOpError("op can have at most one variadic ") << name
          << " specifier";
    if (!std::prev(std::end(vars))->template isa<VariadicT>())
      return op->emitOpError("expected only the last ") << name
          << " to be variadic";
  }
  return success();
}
} // end namespace impl

LogicalResult OperationOp::verify() {
  if (!getAttrOfType<mlir::BoolAttr>(getIsTerminatorAttrName()))
    return emitOpError("expected BoolAttr named: ")
        << getIsTerminatorAttrName();
  if (!getAttrOfType<mlir::BoolAttr>(getIsCommutativeAttrName()))
    return emitOpError("expected BoolAttr named: ")
        << getIsCommutativeAttrName();
  if (!getAttrOfType<mlir::BoolAttr>(getIsIsolatedFromAboveAttrName()))
    return emitOpError("expected BoolAttr named: ")
        << getIsIsolatedFromAboveAttrName();

  auto opTypeAttr = getAttrOfType<mlir::TypeAttr>(getOpTypeAttrName());
  if (!opTypeAttr)
    return emitOpError("expected TypeAttr named: ")
        << getOpTypeAttrName();
  if (!opTypeAttr.getValue().isa<OpType>())
    return emitOpError("expected TypeAttr to contain an `OpType`");

  if (!getAttrOfType<mlir::DictionaryAttr>(getOpAttrDictAttrName()))
    return emitOpError("expected DictionaryAttr named: ")
        << getOpAttrDictAttrName();

  auto opRegions = getAttrOfType<mlir::ArrayAttr>(getOpRegionsAttrName());
  if (!opRegions)
    return emitOpError("expected ArrayAttr named: ")
        << getOpRegionsAttrName();

  auto opSuccs = getAttrOfType<mlir::ArrayAttr>(getOpSuccsAttrName());
  if (!opSuccs)
    return emitOpError("expected ArrayAttr named: ")
        << getOpSuccsAttrName();

  auto opTraits = getAttrOfType<OpTraitsAttr>(getOpTraitsAttrName());
  if (!opTraits)
    return emitOpError("expected OpTraitsAttr named: ")
        << getOpTraitsAttrName();

  /// If there is are variadic values, there must be a size specifier.
  /// More than size specifier is prohobited. Having a size specifier without
  /// variadic values is okay.
  bool hasVariadicArgs = hasVariadicValues(getOpType().getOperandTypes());
  bool hasSameSizedArgs = impl::hasTrait<SameVariadicOperandSizes>(opTraits);
  bool hasArgSegments = impl::hasTrait<SizedOperandSegments>(opTraits);
  if (hasSameSizedArgs && hasArgSegments)
    return emitOpError("cannot have both SameVariadicOperandSizes and ")
        << "SizedOperandSegments traits";
  if (hasVariadicArgs && !hasSameSizedArgs && !hasArgSegments)
    return emitOpError("more than one variadic operands requires a ")
        << "variadic size specifier";

  /// Check results.
  bool hasVariadicRets = hasVariadicValues(getOpType().getResultTypes());
  bool hasSameSizedRets = impl::hasTrait<SameVariadicResultSizes>(opTraits);
  bool hasRetSegments = impl::hasTrait<SizedResultSegments>(opTraits);
  if (hasSameSizedRets && hasRetSegments)
    return emitOpError("cannot have both SameVariadicResultSizes and ")
        << "SizedResultSegments traits";
  if (hasVariadicRets && !hasSameSizedRets && !hasRetSegments)
    return emitOpError("more than one variadic result requires a ")
        << "variadic size specifier";

  /// Verify the region and successor constraint lists.
  if (failed(impl::verifyConstraintList<VariadicRegion>(
          *this, opRegions, &SpecRegion::is, "region")) ||
      failed(impl::verifyConstraintList<VariadicSuccessor>(
          *this, opSuccs, &SpecSuccessor::is, "successor")))
    return failure();

  return success();
}

/// TypeOp.
void TypeOp::build(OpBuilder &builder, OperationState &result,
                   StringRef name, ArrayRef<Attribute> parameters) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute(getParametersAttrName(),
                      builder.getArrayAttr(parameters));
}

// type       ::= `dmc.Type` `@`type-name param-list?
// param-list ::= `<` param (`,` param)* `>`
ParseResult TypeOp::parse(OpAsmParser &parser, OperationState &result) {
  mlir::StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      ParameterList::parse(parser, result.attributes))
    return failure();
  return success();
}

void TypeOp::print(OpAsmPrinter &printer) {
  printer << getOperation()->getName() << ' ';
  printer.printSymbolName(getName());
  printParameterList(printer);
}

/// AttributeOp
void AttributeOp::build(OpBuilder &builder, OperationState &result,
                        StringRef name, ArrayRef<Attribute> parameters) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute(getParametersAttrName(),
                      builder.getArrayAttr(parameters));
}

/// attr ::= `dmc.Attr` `@`attr-name param-list?
ParseResult AttributeOp::parse(OpAsmParser &parser, OperationState &result) {
  mlir::StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      ParameterList::parse(parser, result.attributes))
    return failure();
  return success();
}

void AttributeOp::print(OpAsmPrinter &printer) {
  printer << getOperation()->getName() << ' ';
  printer.printSymbolName(getName());
  printParameterList(printer);
}

/// AliasOp.
void AliasOp::build(OpBuilder &builder, OperationState &result,
                    StringRef name, Type type) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  /// Store the attribute in a TypeAttr.
  result.addAttribute(getAliasedTypeAttrName(), mlir::TypeAttr::get(type));
}

void AliasOp::build(OpBuilder &builder, OperationState &result,
                    StringRef name, Attribute attr) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  /// Store the attribute directly.
  result.addAttribute(getAliasedAttributeAttrName(), attr);

}

ParseResult AliasOp::parse(mlir::OpAsmParser &parser,
                           mlir::OperationState &result) {
  // alias-op ::= `dmc.Alias` `@`symbolname `->` (type|attribute)
  mlir::StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      parser.parseArrow())
    return failure();

  /// Try to parse a type alias.
  Type type;
  auto ret = parser.parseOptionalType(type); // tri-state parsing
  if (ret.hasValue()) {
    if (*ret) // found type but failed to parse
      return failure();
    result.addAttribute(getAliasedTypeAttrName(), mlir::TypeAttr::get(type));
    return success(); // found type and parse succeeded
  }

  /// Parse an attribute alias.
  Attribute attr;
  if (impl::parseSingleAttribute(parser, attr))
    return parser.emitError(parser.getCurrentLocation(),
                            "expected a type or an attribute");
  result.addAttribute(getAliasedAttributeAttrName(), attr);
  return success();
}

void AliasOp::print(OpAsmPrinter &printer) {
  printer << getOperationName() << ' ';
  printer.printSymbolName(getName());
  printer << " -> ";
  if (auto type = getAliasedType()) {
    printer.printType(type);
  } else {
    printer.printAttribute(getAliasedAttr());
  }
}

LogicalResult AliasOp::verify() {
  if (!getAliasedType() && !getAliasedAttr())
    return emitOpError("is neither an alias to a type nor an attribute");
  if (getAliasedType() && getAliasedAttr())
    return emitOpError("cannot be an alias to both a type and an attribute");
  return success();
}

Type AliasOp::getAliasedType() {
  auto typeAlias = getAttrOfType<mlir::TypeAttr>(getAliasedTypeAttrName());
  return typeAlias ? typeAlias.getValue() : Type{};
}

Attribute AliasOp::getAliasedAttr() {
  return getAttr(getAliasedAttributeAttrName()); // returns null if DNE
}

} // end namespace dmc
