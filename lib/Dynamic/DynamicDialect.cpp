#include "dmc/Dynamic/Alias.h"
#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/Dynamic/DynamicType.h"
#include "dmc/Dynamic/DynamicAttribute.h"
#include "dmc/Dynamic/DynamicDialect.h"
#include "dmc/Dynamic/DynamicOperation.h"

#include <mlir/IR/StandardTypes.h>

using namespace mlir;

namespace dmc {

std::unique_ptr<DynamicOperation>
DynamicDialect::createDynamicOp(StringRef name) {
  /// Allocate on heap so AbstractOperation references stay valid.
  /// Ownership must be passed to DynamicContext.
  return std::make_unique<DynamicOperation>(name, this);
}

LogicalResult DynamicDialect::createDynamicType(StringRef name,
                                                NamedParameterRange paramSpec) {
  return registerDynamicType(
      std::make_unique<DynamicTypeImpl>(this, name, paramSpec));
}

LogicalResult DynamicDialect::createDynamicAttr(StringRef name,
                                                NamedParameterRange paramSpec) {
  return registerDynamicAttr(
      std::make_unique<DynamicAttributeImpl>(this, name, paramSpec));
}

Type DynamicDialect::parseType(DialectAsmParser &parser) const {
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  /// Get the type name.
  StringRef name;
  if (parser.parseKeyword(&name))
    return {};

  /// Return a type alias if one is found.
  auto *typeAlias = lookupTypeAlias(name);
  if (typeAlias)
    return typeAlias->getAliasedType();

  /// Lookup a dynamic type and call its parser.
  auto *typeImpl = lookupType(name);
  if (!typeImpl) {
    emitError(loc) << "Unknown type name: " << name;
    return {};
  }
  return typeImpl->parseType(loc, parser);
}

void DynamicDialect::printType(Type type, DialectAsmPrinter &printer) const {
  auto dynTy = type.cast<DynamicType>();
  dynTy.getDynImpl()->printType(type, printer);
}

/// TODO Typed custom attributes. Combining types and attributes requires
/// user-written verification code, which isn't possible until a higher-level
/// language is incorporated.
Attribute DynamicDialect::parseAttribute(DialectAsmParser &parser,
                                         Type type) const {
  if (type && !type.isa<mlir::NoneType>()) {
    parser.emitError(parser.getCurrentLocation(),
                     "typed custom attributes currently unsupported");
    return {};
  }
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());

  /// Get the attribute name.
  StringRef name;
  if (parser.parseKeyword(&name))
    return {};

  /// Return an attribute alias if one is found.
  auto *attrAlias = lookupAttrAlias(name);
  if (attrAlias)
    return attrAlias->getAliasedAttr();

  /// Lookup a dynamic attribute and call its parser.
  auto *attrImpl = lookupAttr(name);
  if (!attrImpl) {
    emitError(loc) << "Unknown attribute name: " << name;
    return {};
  }
  return attrImpl->parseAttribute(loc, parser);
}

void DynamicDialect::printAttribute(Attribute attr,
                                    DialectAsmPrinter &printer) const {
  auto dynAttr = attr.cast<DynamicAttribute>();
  dynAttr.getDynImpl()->printAttribute(attr, printer);
}

} // end namespace dmc
