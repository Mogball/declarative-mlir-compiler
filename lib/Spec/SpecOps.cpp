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

} // end namespace dmc
