#pragma once

#include <mlir/IR/Dialect.h>

namespace dmc {

/// Dynamic dialect underlying class. This class hooks Dialect methods
/// into user-specified functions.
class DynamicDialect : public mlir::Dialect {
public:
  DynamicDialect(llvm::StringRef name, mlir::MLIRContext *ctx);
};

} // end namespace dmc
