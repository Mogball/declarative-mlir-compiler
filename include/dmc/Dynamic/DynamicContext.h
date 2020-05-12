#pragma once

#include "DynamicDialect.h"
#include "DynamicOperation.h"
#include "DynamicType.h"
#include "TypeIDAllocator.h"

namespace dmc {

/// Manages the creation and lifetime of dynamic MLIR objects:
/// Dialects, Operations, Types, and Attributes.
class DynamicContext {
public:
  ~DynamicContext();

  DynamicContext(mlir::MLIRContext *ctx);

  mlir::MLIRContext *getContext() { return ctx; }
  TypeIDAllocator *getTypeIDAlloc() { return typeIdAlloc; }

  /// Create a DynamicDialect and return an instance registered with
  /// the MLIRContext.
  DynamicDialect *createDynamicDialect(llvm::StringRef name);

  /// Register a DynamicOperation with this context so its config
  /// is stored for later use. The context takes ownership.
  void registerDynamicOp(DynamicOperation *op);
  /// Lookup the DynamicOperation belonging to an Operation.
  DynamicOperation *lookupOp(mlir::Operation *op);

  /// Register a DynamicType with the context. The context takes ownership.
  void registerDynamicType(DynamicTypeImpl *type);
  /// Lookup a DynamicType with the given name.
  DynamicTypeImpl *lookupType(llvm::StringRef name);

private:
  class Impl;
  std::unique_ptr<Impl> impl;

  mlir::MLIRContext *ctx;
  TypeIDAllocator *typeIdAlloc;
};

} // end namespace dmc
