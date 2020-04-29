#pragma once

#include <mlir/IR/OperationSupport.h>

#include "DynamicObject.h"

namespace dmc {

/// Forward declarations.
class DynamicDialect;

/// A DynamicTrait captures an invariant about the operation.
class DynamicTrait {
public:
  virtual ~DynamicTrait() = default;
  virtual mlir::LogicalResult verifyOp(mlir::Operation *op) const = 0;
  virtual mlir::AbstractOperation::OperationProperties
  getTraitProperties() const {
    return mlir::AbstractOperation::OperationProperties{};
  }
};

/// This class dynamically captures properties of an Operation.
class DynamicOperation : public DynamicObject {
public:
  DynamicOperation(llvm::StringRef name, DynamicDialect *dialect);

  /// Get the Op representation.
  inline const mlir::AbstractOperation *getOpInfo() {
    assert(opInfo != nullptr && "Op has not been registered");
    return opInfo;
  }

  /// Add a DynamicTrait to this Op. Traits specify invariants on an
  /// Operation checked under verifyInvariants(). OpTraits should be
  /// added only during Op creation.
  void addOpTrait(std::unique_ptr<DynamicTrait> trait);

  /// DynamicOperation creation: define the Base Operation, add properties,
  /// traits, custom functions, hooks, etc, then register with Dialect.
  void finalize();

  /// Delegate function to verify each OpTrait.
  mlir::LogicalResult verifyOpTraits(mlir::Operation *op) const;
  /// Get amalgamated Operation properties from traits.
  mlir::AbstractOperation::OperationProperties getOpProperties() const;

private:
  // Full operation name: `dialect`.`opName`
  const std::string name;
  /// Associated Dialect
  DynamicDialect * const dialect;

  /// A list of OpTraits.
  std::vector<std::unique_ptr<DynamicTrait>> traits;

  // Operation info
  const mlir::AbstractOperation *opInfo;
};

} // end namespace dmc
