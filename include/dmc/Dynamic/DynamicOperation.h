#pragma once

#include "DynamicObject.h"

#include <llvm/ADT/StringMap.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/LoopLikeInterface.h>

namespace dmc {

/// Forward declarations.
class DynamicDialect;

/// A DynamicTrait captures an invariant about the operation.
class DynamicTrait {
public:
  virtual ~DynamicTrait() = default;
  virtual mlir::LogicalResult verifyOp(mlir::Operation *op) const {
    return mlir::success();
  }
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
  /// Get the Op's dialect.
  inline auto *getDialect() const { return dialect; }

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
  llvm::Optional<std::string> parserFcn, printerFcn;

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

/// Mark dynamic operations with this OpTrait. Also, Op requires at least one
/// OpTrait.
template <typename ConcreteType>
class DynamicOpTrait :
    public mlir::OpTrait::TraitBase<ConcreteType, DynamicOpTrait> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    // Hook into the DynamicTraits
    return DynamicOperation::of(op)->verifyOpTraits(op);
  }
};

/// Define base properies of all dynamic ops.
class BaseOp : public mlir::Op<BaseOp, DynamicOpTrait,
               mlir::MemoryEffectOpInterface::Trait,
               mlir::OpTrait::HasRecursiveSideEffects,
               mlir::LoopLikeOpInterface::Trait> {
public:
  using Op::Op;

  // AbstractOperation
  static mlir::ParseResult parseAssembly(mlir::OpAsmParser &parser,
                                         mlir::OperationState &result);

  static void printAssembly(mlir::Operation *op, mlir::OpAsmPrinter &p) {
    // Assume we have the correct Op
    DynamicOperation::of(op)->printOperation(p, op);
  }

  static mlir::LogicalResult verifyInvariants(mlir::Operation *op) {
    // TODO add call to custom verify() function
    // A DynamicOperation will always only have this trait
    return DynamicOpTrait::verifyTrait(op);
  }

  static mlir::LogicalResult foldHook(
      mlir::Operation *op, llvm::ArrayRef<mlir::Attribute> operands,
      llvm::SmallVectorImpl<mlir::OpFoldResult> &results) {
    // TODO custom fold hooks
    return mlir::failure();
  }

  // MemoryEffectOpInterface
  void getEffects(llvm::SmallVectorImpl<mlir::SideEffects::EffectInstance<
                  mlir::MemoryEffects::Effect>> &effects);

  // LoopLikeOpInterface
  mlir::Region &getLoopBody();
  bool isDefinedOutsideOfLoop(mlir::Value value);
  mlir::LogicalResult moveOutOfLoop(llvm::ArrayRef<mlir::Operation *> ops);
  bool canBeHoisted(mlir::Operation *op);

  // Custom isa<T>
  static bool classof(mlir::Operation *op);
};

} // end namespace dmc
