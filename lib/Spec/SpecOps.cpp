#include "dmc/Spec/SpecOps.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>

using namespace mlir;

namespace dmc {

/// DialectOp.
void DialectOp::build(OpBuilder &builder, OperationState &result, 
                      StringRef name) {
  ensureTerminator(*result.addRegion(), builder, result.location);
  result.attributes.push_back(builder.getNamedAttr(
        mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name)));
}

/// dialect-op ::= `Dialect` `@`symbolName `attributes` attr-list region
ParseResult DialectOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  auto *body = result.addRegion();
  if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(), 
                             result.attributes) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes) || 
      parser.parseRegion(*body, llvm::None, llvm::None))
    return failure();
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
  // TODO verify dialect attributes once they are added to DynamicDialect
  return success();
}

Region &DialectOp::getBodyRegion() { return getOperation()->getRegion(0); }
Block *DialectOp::getBody() { return &getBodyRegion().front(); }

StringRef DialectOp::getName() {
  return getAttrOfType<StringAttr>(mlir::SymbolTable::getSymbolAttrName())
      .getValue();
}

/// OperationOp
void OperationOp::build(OpBuilder &builder, OperationState &result,
                        StringRef name, FunctionType type,
                        ArrayRef<NamedAttribute> attrs) {
  result.addAttribute(SymbolTable::getSymbolAttrName(),
                      builder.getStringAttr(name));
  result.addAttribute(getTypeAttrName(), TypeAttr::get(type));
  result.attributes.append(std::begin(attrs), std::end(attrs));
  result.addRegion();
}

template <typename ContainerT>
static ParseResult parseTypeList(OpAsmParser &parser, ContainerT &types) {
  if (parser.parseLParen())
    return failure();
  Type ty;
  auto ret = parser.parseOptionalType(ty);
  if (ret.hasValue()) {
    if (ret.getValue()) 
      return failure();
    types.push_back(ty);
    while (!parser.parseOptionalComma()) {
      if (parser.parseType(ty))
        return failure();
      types.push_back(ty);
    }
  }
  if (parser.parseRParen())
    return failure();
  return success();
}

static void printTypeList(OpAsmPrinter &printer, ArrayRef<Type> types) {
  printer << '(';
  auto it = std::begin(types), e = std::end(types);
  if (it != e) {
    printer.printType(*it);
    for (; ++it != e;) {
      printer << ',';
      printer.printType(*it);
    }
  }
  printer << ')';
}

// op ::= `dmc.Op` `@`opName type-list `->` type-list `attributes` attr-list 
ParseResult OperationOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  SmallVector<Type, 4> argTys;
  SmallVector<Type, 2> retTys;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(), 
                             result.attributes) ||
      parseTypeList(parser, argTys) || parser.parseArrow() ||
      parseTypeList(parser, retTys) || 
      parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();
  auto funcType = FunctionType::get(argTys, retTys, result.getContext()); 
  result.addAttribute(getTypeAttrName(), TypeAttr::get(funcType));
  result.addRegion();
  return success();
}

void OperationOp::print(OpAsmPrinter &printer) {
  printer << getOperation()->getName() << ' ';
  printer.printSymbolName(getName());
  auto funcType = getType();
  printTypeList(printer, funcType.getInputs());
  printer << " -> ";
  printTypeList(printer, funcType.getResults());
  printer.printOptionalAttrDictWithKeyword(getAttrs(), {
      SymbolTable::getSymbolAttrName(), getTypeAttrName()});
}

StringRef OperationOp::getName() {
  return getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
      .getValue();
}

LogicalResult OperationOp::verify() {
  // TODO verify variadic and optional arg conformance
  // TODO verify configs
  return success();
}

LogicalResult OperationOp::verifyType() {
  auto type = getTypeAttr().getValue();
  if (!type.isa<FunctionType>())
    return emitOpError("requires '" + getTypeAttrName() + 
                       "' atttribute of function type");
  return success();
}

} // end namespace dmc
