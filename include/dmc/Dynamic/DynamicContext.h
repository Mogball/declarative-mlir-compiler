#pragma once

#include "TypeIDAllocator.h"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Attributes.h>

namespace dmc {

/// Forward declarations.
class DynamicDialect;

/// Manages the creation and lifetime of dynamic MLIR objects:
/// Dialects, Operations, Types, and Attributes.
class DynamicContext {
public:
  ~DynamicContext();
  DynamicContext(mlir::MLIRContext *ctx);

  /// Getters.
  mlir::MLIRContext *getContext() { return ctx; }
  TypeIDAllocator *getTypeIDAlloc() { return typeIdAlloc; }

  /// Create a DynamicDialect and return an instance registered with
  /// the MLIRContext.
  DynamicDialect *createDynamicDialect(llvm::StringRef name);
  /// Lookup the dynamic dialect belonging to a dynamic MLIR object. This is
  /// necessary since aliased types and attributes do subclass a generic class.
  ///
  /// TODO This is not an ideal solution.
  DynamicDialect *lookupDialectFor(mlir::Type type);
  DynamicDialect *lookupDialectFor(mlir::Attribute attr);
  mlir::LogicalResult registerDialectSymbol(DynamicDialect *dialect,
                                            mlir::Type type);
  mlir::LogicalResult registerDialectSymbol(DynamicDialect *dialect,
                                            mlir::Attribute attr);

private:
  class Impl;
  mlir::MLIRContext *ctx;
  TypeIDAllocator *typeIdAlloc;
  std::unique_ptr<Impl> impl;
};

} // end namespace dmc
