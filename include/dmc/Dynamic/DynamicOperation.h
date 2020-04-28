#pragma once

#include <mlir/IR/OperationSupport.h>

#include "DynamicObject.h"

namespace dmc {

/// Forward declarations.
class DynamicDialect;

/// This class dynamically captures properties of an Operation.
class DynamicOperation : public DynamicObject {
public:
  DynamicOperation(llvm::StringRef name, DynamicDialect *dialect);

  /// Get the Op representation.
  inline mlir::AbstractOperation getOpInfo() { return opInfo; }

private:
  // Full operation name: `dialect`.`opName`
  std::string name;

  mlir::AbstractOperation opInfo;
};

} // end namespace dmc
