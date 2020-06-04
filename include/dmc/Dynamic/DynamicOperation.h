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

  /// Get the Op name.
  inline auto &getName() const { return name; }

  /// Add a DynamicTrait to this Op. Traits specify invariants on an
  /// Operation checked under verifyInvariants(). OpTraits should be
  /// added only during Op creation.
  mlir::LogicalResult addOpTrait(llvm::StringRef name,
                                 std::unique_ptr<DynamicTrait> trait);
  template <typename TraitT, typename... Args>
  mlir::LogicalResult addOpTrait(Args &&... args);

  /// Set a custom parser and printer.
  void setOpFormat(std::string parserName, std::string printerName);

  /// DynamicOperation creation: define the Base Operation, add properties,
  /// traits, custom functions, hooks, etc, then register with Dialect.
  ///
  /// Returns failure() if another Operation with the same name exists.
  mlir::LogicalResult finalize();

  /// Delegate function to verify each OpTrait.
  mlir::LogicalResult verifyOpTraits(mlir::Operation *op) const;
  /// Get amalgamated Operation properties from traits.
  mlir::AbstractOperation::OperationProperties getOpProperties() const;

  /// Higher-level DynamicOperation specification info is made
  /// available to traits and other verifiers through traits.
  template <typename TraitT> TraitT *getTrait();
  DynamicTrait *getTrait(llvm::StringRef);

  /// Parse or print an operation.
  mlir::ParseResult parseOperation(mlir::OpAsmParser &parser,
                                   mlir::OperationState &result);
  void printOperation(mlir::OpAsmPrinter &printer, mlir::Operation *op);

private:
  /// Full operation name: `dialect`.`opName`.
  const std::string name;
  /// Associated Dialect.
  DynamicDialect * const dialect;

  /// A list of dynamic OpTraits. Lookups and insertions are linear time as
  /// there are assumed to be few traits. Using a vector also guarantees that
  /// the traits are checked in insertion order.
  std::vector<std::pair<llvm::StringRef, std::unique_ptr<DynamicTrait>>> traits;

  /// The function names of the custom parser and printers, if present.
  std::optional<std::string> parserFcn, printerFcn;

  // Operation info
  const mlir::AbstractOperation *opInfo;
};

/// Out-of-line definitions
template <typename TraitT, typename... Args>
mlir::LogicalResult DynamicOperation::addOpTrait(Args &&... args) {
  return addOpTrait(TraitT::getName(), std::make_unique<TraitT>(
      std::forward<Args>(args)...));
}

template <typename TraitT> TraitT *DynamicOperation::getTrait() {
  return dynamic_cast<TraitT *>(getTrait(TraitT::getName()));
}

} // end namespace dmc
