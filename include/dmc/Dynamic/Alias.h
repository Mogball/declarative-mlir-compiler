#pragma once

#include "Metadata.h"

#include <mlir/IR/Attributes.h>

namespace dmc {

/// An alias allows shorthand definitions of types or attributes.
/// See `dmc.Alias`. Aliases are erased during parsing of the dialect module.
///
/// `DynamicDialect::parseType`, for example, upon parsing a type alias, return
/// the aliased type.
class TypeAlias : public TypeMetadata {
public:
  TypeAlias(llvm::StringRef name, mlir::Type aliasedType,
            llvm::Optional<llvm::StringRef> builder = {})
      : TypeMetadata{name, builder},
        aliasedType{aliasedType} {}

  inline auto getAliasedType() { return aliasedType; }

private:
  mlir::Type aliasedType;
};

/// `DynamicDialect::parseAttribute` will directly return the aliased
/// attribute.
class AttributeAlias : public AttributeMetadata {
public:
  AttributeAlias(llvm::StringRef name, mlir::Attribute aliasedAttr,
                 llvm::Optional<llvm::StringRef> builder = {},
                 mlir::Type type = {})
      : AttributeMetadata{name, builder, type},
        aliasedAttr{aliasedAttr} {}

  inline auto getAliasedAttr() { return aliasedAttr; }

private:
  mlir::Attribute aliasedAttr;
};

} // end namespace dmc
