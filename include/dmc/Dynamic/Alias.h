#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Types.h>

namespace dmc {

/// An alias allows shorthand definitions of types or attributes.
/// See `dmc.Alias`. Aliases are erased during parsing of the dialect module.
///
/// `DynamicDialect::parseType`, for example, upon parsing a type alias, return
/// the aliased type.
class TypeAlias {
public:
  TypeAlias(llvm::StringRef name, mlir::Type aliasedType)
      : name{name},
        aliasedType{aliasedType} {}

  inline auto getName() { return name; }
  inline auto getAliasedType() { return aliasedType; }

private:
  llvm::StringRef name;
  mlir::Type aliasedType;
};

/// `DynamicDialect::parseAttribute` will directly return the aliased
/// attribute.
class AttributeAlias {
public:
  AttributeAlias(llvm::StringRef name, mlir::Attribute aliasedAttr)
      : name{name},
        aliasedAttr{aliasedAttr} {}

  inline auto getName() { return name; }
  inline auto getAliasedAttr() { return aliasedAttr; }

private:
  llvm::StringRef name;
  mlir::Attribute aliasedAttr;
};

} // end namespace dmc
