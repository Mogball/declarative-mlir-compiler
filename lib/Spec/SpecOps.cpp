#include "dmc/Spec/SpecOps.h"
#include "dmc/Spec/SpecTypes.h"
#include "dmc/Spec/SpecAttrs.h"
#include "dmc/Traits/SpecTraits.h"
#include "dmc/Traits/Registry.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>

using namespace mlir;

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

StringRef DialectOp::getName() {
  return getAttrOfType<mlir::StringAttr>(mlir::SymbolTable::getSymbolAttrName())
      .getValue();
}

bool DialectOp::allowsUnknownOps() {
  return getAttrOfType<mlir::BoolAttr>(getAllowUnknownOpsAttrName())
      .getValue();
}

bool DialectOp::allowsUnknownTypes() {
  return getAttrOfType<mlir::BoolAttr>(getAllowUnknownTypesAttrName())
      .getValue();
}

/// OperationOp
void OperationOp::setOpType(mlir::FunctionType opTy) {
  auto newType = mlir::TypeAttr::get(opTy);
  setAttr(getTypeAttrName(), mlir::TypeAttr::get(opTy));
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

void OperationOp::build(OpBuilder &builder, OperationState &result,
                        StringRef name, mlir::FunctionType type,
                        ArrayRef<NamedAttribute> attrs) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute(getTypeAttrName(), mlir::TypeAttr::get(type));
  result.addAttribute(getOpTraitsAttrName(), builder.getDictionaryAttr(attrs));
  result.addRegion();
  buildDefaultValuedAttrs(builder, result);
}

// op ::= `dmc.Op` `@`opName type-list `->` type-list `attributes` attr-list
//        `config` attr-list
ParseResult OperationOp::parse(OpAsmParser &parser, OperationState &result) {
  mlir::StringAttr nameAttr;
  mlir::TypeAttr funcTypeAttr;
  NamedAttrList opAttrs;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      parser.parseAttribute(funcTypeAttr, getTypeAttrName(),
                            result.attributes) ||
      parser.parseOptionalAttrDict(opAttrs))
    return failure();
  result.addAttribute(getOpAttrDictAttrName(),
                      mlir::DictionaryAttr::get(opAttrs, result.getContext()));
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
  printer.printSymbolName(getName().getValue());
  printer.printType(getType());
  printer << ' ';
  printer.printAttribute(getOpAttrs());
  printer << " config";
  printer.printOptionalAttrDict(getAttrs(), {
      SymbolTable::getSymbolAttrName(), getTypeAttrName(),
      getOpAttrDictAttrName()});
}

mlir::StringAttr OperationOp::getName() {
  return getAttrOfType<mlir::StringAttr>(SymbolTable::getSymbolAttrName());
}

mlir::DictionaryAttr OperationOp::getOpAttrs() {
  return getAttrOfType<mlir::DictionaryAttr>(getOpAttrDictAttrName());
}

mlir::ArrayAttr OperationOp::getOpTraits() {
  return getAttrOfType<mlir::ArrayAttr>(getOpTraitsAttrName());
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

/// Trait array manipulation helpers.
namespace impl {

template <typename TraitT>
bool hasTrait(mlir::ArrayAttr opTraits) {
  return llvm::count_if(opTraits.getAsRange<mlir::FlatSymbolRefAttr>(),
      [](mlir::FlatSymbolRefAttr sym) {
          return sym.getValue() == TraitT::getName();
      });
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
  auto opTraits = getAttrOfType<mlir::ArrayAttr>(getOpTraitsAttrName());
  if (!opTraits)
    return emitOpError("expected ArrayAttr named: ")
        << getOpTraitsAttrName();
  /// TODO support full SymbolRefAttr to refer to dynamic traits, e.g.
  /// `@Python.MyOpTrait`
  else if (llvm::count_if(opTraits,
        [](Attribute attr) { return !attr.isa<mlir::FlatSymbolRefAttr>(); }))
    return emitOpError("expected ArrayAttr '") << getOpTraitsAttrName()
        << "' of only FlatSymbolRefAttr";

  /// If there is are variadic values, there must be a size specifier.
  /// More than size specifier is prohobited. Having a size specifier without
  /// variadic values is okay.
  bool hasVariadicArgs = hasVariadicValues(getType().getInputs());
  bool hasSameSizedArgs = impl::hasTrait<SameVariadicOperandSizes>(opTraits);
  bool hasArgSegments = impl::hasTrait<SizedOperandSegments>(opTraits);
  if (hasSameSizedArgs && hasArgSegments)
    return emitOpError("cannot have both SameVariadicOperandSizes and ")
        << "SizedOperandSegments traits";
  if (hasVariadicArgs && !hasSameSizedArgs && !hasArgSegments)
    return emitOpError("more than one variadic operands requires a ")
        << "variadic size specifier";
  /// Check results.
  bool hasVariadicRets = hasVariadicValues(getType().getResults());
  bool hasSameSizedRets = impl::hasTrait<SameVariadicResultSizes>(opTraits);
  bool hasRetSegments = impl::hasTrait<SizedResultSegments>(opTraits);
  if (hasSameSizedRets && hasRetSegments)
    return emitOpError("cannot have both SameVariadicResultSizes and ")
        << "SizedResultSegments traits";
  if (hasVariadicRets && !hasSameSizedRets && !hasRetSegments)
    return emitOpError("more than one variadic result requires a ")
        << "variadic size specifier";

  /// Verify that the remaining traits exist
  auto *registry = getContext()->getRegisteredDialect<TraitRegistry>();
  assert(registry != nullptr && "TraitRegistry dialect was not registered");
  for (auto trait : opTraits.getAsRange<mlir::FlatSymbolRefAttr>()) {
    if (!registry->lookupTrait(trait.getValue()))
      return emitOpError("trait '") << trait.getValue() << "' not found";
  }
  return success();
}

LogicalResult OperationOp::verifyType() {
  if (!getType().isa<mlir::FunctionType>())
    return emitOpError("requires '" + getTypeAttrName() +
                       "' attribute of function type");
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

/// Parse a single attribute with OpAsmParser. OpAsmParser only provides
/// a function to parse a named attribute into a list.
namespace impl {
ParseResult parseSingleAttribute(Attribute &attr, OpAsmParser &parser) {
  constexpr auto attrName = "attr";
  NamedAttrList attrList;
  return parser.parseAttribute(attr, attrName, attrList);
}
} // end namespace impl

// type       ::= `dmc.Type` `@`type-name param-list?
// param-list ::= `<` param (`,` param)* `>`
ParseResult TypeOp::parse(OpAsmParser &parser, OperationState &result) {
  mlir::StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  SmallVector<Attribute, 2> parameters;
  if (!parser.parseOptionalLess()) {
    do {
      Attribute parameter;
      if (impl::parseSingleAttribute(parameter, parser))
        return failure();
      // If a type attribute was specified, create an OfTypeAttr
      if (auto typeAttr = parameter.dyn_cast<mlir::TypeAttr>())
        parameter = OfTypeAttr::get(typeAttr.getValue());
      parameters.push_back(parameter);
    } while (!parser.parseOptionalComma());
    if (parser.parseGreater())
      return failure();
  }
  result.addAttribute(getParametersAttrName(),
      mlir::ArrayAttr::get(parameters, result.getContext()));
  return success();
}

void TypeOp::print(OpAsmPrinter &printer) {
  printer << getOperation()->getName() << ' ';
  printer.printSymbolName(getName());
  auto parameters = getParameters();
  if (!parameters.empty()) {
    auto it = std::begin(parameters);
    printer << '<' << (*it++);
    for (auto e = std::end(parameters); it != e; ++it)
      printer << ',' << (*it);
    printer << '>';
  }
}

StringRef TypeOp::getTypeName() {
  return getAttrOfType<mlir::StringAttr>(SymbolTable::getSymbolAttrName())
      .getValue();
}

ArrayRef<Attribute> TypeOp::getParameters() {
  return getAttrOfType<mlir::ArrayAttr>(getParametersAttrName())
      .getValue();
}

LogicalResult TypeOp::verify() {
  if (!getAttrOfType<mlir::StringAttr>(SymbolTable::getSymbolAttrName()))
    return emitOpError("expected StringAttr named: ")
        << SymbolTable::getSymbolAttrName();
  if (!getAttrOfType<mlir::ArrayAttr>(getParametersAttrName()))
    return emitOpError("expected ArrayAttr named: ")
        << getParametersAttrName();

  /// Check that all attributes are SpecAttr.
  unsigned idx = 0;
  for (auto it = std::begin(getParameters()), e = std::end(getParameters());
       it != e; ++it, ++idx) {
    if (!SpecAttrs::is(*it))
      return emitOpError("expected parameter #") << idx << " to be an "
          << "attribute constraint, but got " << *it;
  }
  return success();
}

} // end namespace dmc
