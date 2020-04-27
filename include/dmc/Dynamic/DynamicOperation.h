#pragma once

#include <mlir/IR/OperationSupport.h>

#include "DynamicObject.h"

namespace dmc {

/// Forward declarations.
class DynamicDialect;

class DynamicOperation : public DynamicObject {
public:
  DynamicOperation(llvm::StringRef name, DynamicDialect *dialect);

private:
  mlir::AbstractOperation opInfo;
};

} // end namespace dmc
