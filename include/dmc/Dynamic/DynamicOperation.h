#pragma once

#include <mlir/IR/OperationSupport.h>

#include "DynamicObject.h"

namespace dmc {

/// Forward declarations.
class DynamicDialect;

/// A DynamicTrait captures an invariant about the operation.
class DynamicTraitBase {
public:
  /// DynamicTraits are distinguished by a kind.
  inline explicit DynamicTraitBase(unsigned kind) : kind{kind} {}

  virtual ~DynamicTraitBase() = default;
  virtual mlir::LogicalResult verifyOp(mlir::Operation *op) const = 0;
  virtual mlir::AbstractOperation::OperationProperties
  getTraitProperties() const {
    return mlir::AbstractOperation::OperationProperties{};
  }

  /// Get the trait kind.
  inline unsigned getKind() { return kind; }

private:
  unsigned kind;
};

template <unsigned Kind>
class DynamicTrait : public DynamicTraitBase {
public:
  static constexpr auto kind = Kind;

  inline DynamicTrait() : DynamicTraitBase{Kind} {}
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
  void addOpTrait(std::unique_ptr<DynamicTraitBase> trait);

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
  std::vector<std::unique_ptr<DynamicTraitBase>> traits;

  // Operation info
  const mlir::AbstractOperation *opInfo;
};

/// Out-of-line definitions
template <typename TraitT>
TraitT *DynamicOperation::getTrait() {
  /// TODO for "hard" traits, use standard MLIR OpTraits?
  for (auto &trait : traits) {
    if (TraitT::kind == trait->getKind())
      return static_cast<TraitT *>(trait.get());
  }
  return nullptr;
}

} // end namespace dmc
