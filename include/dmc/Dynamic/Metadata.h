#pragma once

#include <mlir/IR/Types.h>

/// A structure to store metadata for dynamic types and attributes, such as
/// constant builder strings and names. This is to provide a unified source of
/// metadata across aliases and concrete types.
///
/// The metadata will have to be queried directly from the dynamic dialect.
namespace dmc {

class TypeMetadata {
public:
  explicit TypeMetadata(llvm::StringRef name,
                        llvm::Optional<llvm::StringRef> builder)
      : name{name},
        builder{builder} {}

  inline auto getName() { return name; }
  inline auto getBuilder() { return builder; }

private:
  /// The name of the type.
  llvm::StringRef name;
  /// An optional Python builder.
  llvm::Optional<llvm::StringRef> builder;
};

class AttributeMetadata {
public:
  explicit AttributeMetadata(llvm::StringRef name,
                             llvm::Optional<llvm::StringRef> builder,
                             mlir::Type type)
      : name{name},
        builder{builder},
        type{type} {}

  inline auto getName() { return name; }
  inline auto getBuilder() { return builder; }
  inline auto getType() { return type; }

private:
  /// The name of the attribute.
  llvm::StringRef name;
  /// An optional Python builder.
  llvm::Optional<llvm::StringRef> builder;
  /// An optional attribute type.
  mlir::Type type;
};

} // end namespace dmc
