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
  DynamicContext(mlir::MLIRContext *ctx);

  /// Getters.
  mlir::MLIRContext *getContext() { return ctx; }
  TypeIDAllocator *getTypeIDAlloc() { return typeIdAlloc; }

  /// Create a DynamicDialect and return an instance registered with
  /// the MLIRContext.
  DynamicDialect *createDynamicDialect(llvm::StringRef name);

private:
  mlir::MLIRContext *ctx;
  TypeIDAllocator *typeIdAlloc;
};

} // end namespace dmc
