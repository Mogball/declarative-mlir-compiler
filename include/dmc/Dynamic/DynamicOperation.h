#pragma once

#include <mlir/IR/OperationSupport.h>

#include "DynamicObject.h"

namespace dmc {

class DynamicOperation : public DynamicObject {
public:
  DynamicOperation(llvm::StringRef name, DynamicContext *ctx);

private:
  mlir::AbstractOperation opInfo;
};

} // end namespace dmc
