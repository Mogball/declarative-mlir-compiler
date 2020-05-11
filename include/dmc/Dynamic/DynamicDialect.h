#pragma once

#include "DynamicObject.h"

#include <mlir/IR/Dialect.h>

namespace dmc {

/// Forward declarations.
class DynamicContext;
class DynamicOperation;
class DynamicTypeImpl;

/// Dynamic dialect underlying class. This class hooks Dialect methods
/// into user-specified functions.
class DynamicDialect : public mlir::Dialect,
                       public DynamicObject {
public:
  DynamicDialect(llvm::StringRef name, DynamicContext *ctx);

  /// Create a new Op associated with this Dialect. Additional configs are
  /// added directly to the returned DynamicOperation before it is finalized.
  DynamicOperation *createDynamicOp(llvm::StringRef name);

  /// Create a dynamic type with the given name and parameter spec and add
  /// it to the given dialect.
  DynamicTypeImpl *createDynamicType(
      llvm::StringRef name, llvm::ArrayRef<mlir::Attribute> paramSpec);

  /// Expose configuration methods.
  inline void allowUnknownOperations(bool allow) {
    Dialect::allowUnknownOperations(allow);
  }
  inline void allowUnknownTypes(bool allow) {
    Dialect::allowUnknownTypes(allow);
  }

private:
  friend class DynamicOperation;
};

} // end namespace dmc
