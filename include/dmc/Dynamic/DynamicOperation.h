#pragma once

#include <mlir/IR/OperationSupport.h>

#include "DynamicObject.h"

namespace dmc {

/// Forward declarations.
class DynamicDialect;

/// A DynamicTrait captures an invariant about the operation.
class DynamicTrait {
public:
  virtual mlir::LogicalResult verifyOp(mlir::Operation *op) const = 0;
  virtual mlir::AbstractOperation::OperationProperties 
  getTraitProperties() const = 0;
};

/// This class dynamically captures properties of an Operation.
class DynamicOperation : public DynamicObject {
public:
  DynamicOperation(llvm::StringRef name, DynamicDialect *dialect);

  /// Get the Op representation.
  inline const mlir::AbstractOperation *getOpInfo() { return opInfo; }

  /// Add a DynamicTrait to this Op. Traits specify invariants on an 
  /// Operation checked under verifyInvariants(). OpTraits should be
  /// added only during Op creation.
  void addOpTrait(std::unique_ptr<DynamicTrait> trait);
  /// Delegate function to verify each OpTrait.
  mlir::LogicalResult verifyOpTraits(mlir::Operation *op);

private:
  // Full operation name: `dialect`.`opName`
  std::string name;

  // Operation info
  const mlir::AbstractOperation *opInfo;

  /// A list of OpTraits.
  std::vector<std::unique_ptr<DynamicTrait>> traits;
};

} // end namespace dmc
