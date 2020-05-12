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
  ~DynamicDialect();
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

  /// Printing and parsing for dynamic types.
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
  void printType(mlir::Type type,
                 mlir::DialectAsmPrinter &printer) const override;

  /// Register a DynamicOperation with this dialect so its config
  /// is stored for later use. The dialect takes ownership.
  void registerDynamicOp(DynamicOperation *op);
  /// Lookup the DynamicOperation belonging to an Operation.
  DynamicOperation *lookupOp(mlir::Operation *op) const;

  /// Register a DynamicType with the dialect. The dialect takes ownership.
  void registerDynamicType(DynamicTypeImpl *type);
  /// Lookup a DynamicType with the given name.
  DynamicTypeImpl *lookupType(llvm::StringRef name) const;

private:
  class Impl;
  std::unique_ptr<Impl> impl;

  friend class DynamicOperation;
};

} // end namespace dmc
