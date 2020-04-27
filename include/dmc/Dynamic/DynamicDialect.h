#pragma once

#include <mlir/IR/Dialect.h>

#include "DynamicObject.h"

namespace dmc {

/// Forward declarations.
class DynamicContext;

/// Dynamic dialect underlying class. This class hooks Dialect methods
/// into user-specified functions.
class DynamicDialect : public mlir::Dialect,
                       public DynamicObject {
public:
  DynamicDialect(llvm::StringRef name, DynamicContext *ctx);
};

} // end namespace dmc
