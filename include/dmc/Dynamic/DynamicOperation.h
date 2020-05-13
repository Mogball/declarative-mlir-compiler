#pragma once

#include "DynamicObject.h"

#include <llvm/ADT/StringMap.h>
#include <mlir/IR/OperationSupport.h>

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
  /// Lookup the DynamicOperation backing an Operation.
  static DynamicOperation *of(mlir::Operation *op);

  DynamicOperation(llvm::StringRef name, DynamicDialect *dialect);

  /// Get the Op representation.
  inline const mlir::AbstractOperation *getOpInfo() {
    assert(opInfo != nullptr && "Op has not been registered");
    return opInfo;
  }

  /// Add a DynamicTrait to this Op. Traits specify invariants on an
  /// Operation checked under verifyInvariants(). OpTraits should be
  /// added only during Op creation.
  void addOpTrait(llvm::StringRef name,
                  std::unique_ptr<DynamicTrait> trait);
  template <typename TraitT, typename... Args>
  void addOpTrait(Args &&... args);

  /// DynamicOperation creation: define the Base Operation, add properties,
  /// traits, custom functions, hooks, etc, then register with Dialect.
  void finalize();

  /// Delegate function to verify each OpTrait.
  mlir::LogicalResult verifyOpTraits(mlir::Operation *op) const;
  /// Get amalgamated Operation properties from traits.
  mlir::AbstractOperation::OperationProperties getOpProperties() const;

  /// Higher-level DynamicOperation specification info is made
  /// available to traits and other verifiers through traits.
  template <typename TraitT> TraitT *getTrait();

private:
  /// Full operation name: `dialect`.`opName`.
  const std::string name;
  /// Associated Dialect.
  DynamicDialect * const dialect;

  /// A list of dynamic OpTraits.
  llvm::StringMap<std::unique_ptr<DynamicTrait>> traits;

  // Operation info
  const mlir::AbstractOperation *opInfo;
};

/// Out-of-line definitions
template <typename TraitT, typename... Args>
void DynamicOperation::addOpTrait(Args &&... args) {
  addOpTrait(TraitT::getName(), std::make_unique<TraitT>(
        std::forward<Args>(args)...));
}

template <typename TraitT>
TraitT *DynamicOperation::getTrait() {
  /// TODO for "hard" traits, use standard MLIR OpTraits with DynamicOperation?
  auto it = traits.find(TraitT::getName());
  if (it == std::end(traits))
    return nullptr;
  return dynamic_cast<TraitT *>(it->second.get());
}

} // end namespace dmc
