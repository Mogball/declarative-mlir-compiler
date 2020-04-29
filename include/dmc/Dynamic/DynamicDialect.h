#pragma once

#include <mlir/IR/Dialect.h>

#include "DynamicObject.h"

namespace dmc {

/// Forward declarations.
class DynamicContext;
class DynamicOperation;

/// Dynamic dialect underlying class. This class hooks Dialect methods
/// into user-specified functions.
class DynamicDialect : public mlir::Dialect,
                       public DynamicObject {
public:
  DynamicDialect(llvm::StringRef name, DynamicContext *ctx);

  /// Create a new Op associated with this Dialect. Additional configs are
  /// added directly to the returned DynamicOperation before it is finalized.
  DynamicOperation *createDynamicOp(llvm::StringRef name);

private:
  friend class DynamicOperation;
};

} // end namespace dmc
