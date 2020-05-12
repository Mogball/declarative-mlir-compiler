#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/Dynamic/DynamicDialect.h"
#include "dmc/Dynamic/DynamicOperation.h"

using namespace mlir;

namespace dmc {

DynamicOperation *DynamicDialect::createDynamicOp(StringRef name) {
  /// Allocate on heap so AbstractOperation references stay valid.
  /// Ownership must be passed to DynamicContext.
  return new DynamicOperation(name, this);
}

DynamicTypeImpl *DynamicDialect::createDynamicType(
    StringRef name, ArrayRef<Attribute> paramSpec) {
  auto *type = new DynamicTypeImpl(this, name, paramSpec);
  /// Take ownership of the type.
  registerDynamicType(type);
  return type;
}

Type DynamicDialect::parseType(DialectAsmParser &parser) const {
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  StringRef name;
  if (parser.parseKeyword(&name))
    return Type{};
  auto *typeImpl = lookupType(name);
  if (!typeImpl) {
    emitError(loc) << "Unknown type name: " << name;
    return Type{};
  }
  return typeImpl->parseType(loc, parser);
}

void DynamicDialect::printType(Type type, DialectAsmPrinter &printer) const {
  auto dynTy = type.cast<DynamicType>();
  dynTy.getTypeImpl()->printType(type, printer);
}

} // end namespace dmc
