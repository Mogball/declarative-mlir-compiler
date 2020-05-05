#pragma once

#include "Kinds.h"
#include "dmc/Dynamic/DynamicOperation.h"

/// Bind common OpTraits into DynamicTraits
namespace dmc {

template <template <typename ConcreteType> class BaseTrait>
struct BindTrait : public DynamicTrait {
  mlir::LogicalResult verifyOp(mlir::Operation *op) const override {
    // Provide dummy template arg
    return BaseTrait<DynamicTrait>::verifyTrait(op);
  }

  mlir::AbstractOperation::OperationProperties
  getTraitProperties() const override {
    return BaseTrait<DynamicTrait>::getTraitProperties();
  }
};
template <typename Arg>
class BindArgTrait : public DynamicTrait {
protected:
  using ArgTy = Arg;
  using TraitImpl = typename
    std::add_pointer<mlir::LogicalResult(mlir::Operation *, ArgTy)>::type;

public:
  BindArgTrait(TraitImpl impl, Arg arg) : impl{impl}, arg{arg} {};

  mlir::LogicalResult verifyOp(mlir::Operation *op) const override {
    return impl(op, arg);
  }

private:
  TraitImpl impl;
  ArgTy arg;
};

/// Simple traits can be rebound.
struct IsTerminator
    : public BindTrait<mlir::OpTrait::IsTerminator> {
  static llvm::StringRef getName() { return "IsTerminator"; }
};
struct IsCommutative
    : public BindTrait<mlir::OpTrait::IsCommutative> {
  static llvm::StringRef getName() { return "IsCommutative"; }
};
struct IsIsolatedFromAbove
    : public BindTrait<mlir::OpTrait::IsIsolatedFromAbove> {
  static llvm::StringRef getName() { return "IsIsolatedFromAbove"; }
};

struct OperandsAreFloatLike
    : public BindTrait<mlir::OpTrait::OperandsAreFloatLike> {
  static llvm::StringRef getName() { return "OperandsAreFloatLike"; }
};
struct OperandsAreSignlessIntegerLike
    : public BindTrait<mlir::OpTrait::OperandsAreSignlessIntegerLike> {
  static llvm::StringRef getName() { return "OperandsAreSignlessIntegerLike"; }
};
struct ResultsAreBoolLike
    : public BindTrait<mlir::OpTrait::ResultsAreBoolLike> {
  static llvm::StringRef getName() { return "ResultsAreBoolLike"; }
};
struct ResultsAreFloatLike
    : public BindTrait<mlir::OpTrait::ResultsAreFloatLike> {
  static llvm::StringRef getName() { return "ResultsAreFloatLike"; }
};
struct ResultsAreSignlessIntegerLike
    : public BindTrait<mlir::OpTrait::ResultsAreSignlessIntegerLike> {
  static llvm::StringRef getName() { return "ResultsAreSignlessIntegerLike"; }
};

struct SameOperandsShape
    : public BindTrait<mlir::OpTrait::SameOperandsShape> {
  static llvm::StringRef getName() { return "SameOperandsShape"; }
};
struct SameOperandsAndResultShape
    : public BindTrait<mlir::OpTrait::SameOperandsAndResultShape> {
  static llvm::StringRef getName() { return "SameOperandsAndResultShape"; }
};
struct SameOperandsElementType
    : public BindTrait<mlir::OpTrait::SameOperandsElementType> {
  static llvm::StringRef getName() { return "SameOperandsElementType"; }
};
struct SameOperandsAndResultElementType
    : public BindTrait<mlir::OpTrait::SameOperandsAndResultElementType> {
  static llvm::StringRef getName() { return "SameOperandsAndResultElementType"; }
};
struct SameOperandsAndResultType
    : public BindTrait<mlir::OpTrait::SameOperandsAndResultType> {
  static llvm::StringRef getName() { return "SameOperandsAndResultType"; }
};
struct SameTypeOperands
    : public BindTrait<mlir::OpTrait::SameTypeOperands> {
  static llvm::StringRef getName() { return "SameTypeOperands"; }
};

/// Stateful traits require constructors.
struct NOperands : public BindArgTrait<unsigned> {
  static llvm::StringRef getName() { return "NOperands"; }

  explicit NOperands(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyNOperands, num) {}
};
struct AtLeastNOperands : public BindArgTrait<unsigned> {
  static llvm::StringRef getName() { return "AtLeastNOperands"; }

  explicit AtLeastNOperands(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyAtLeastNOperands, num) {}
};
struct NRegions : public BindArgTrait<unsigned> {
  static llvm::StringRef getName() { return "NRegions"; }

  explicit NRegions(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyNRegions, num) {}
};
struct AtLeastNRegions : public BindArgTrait<unsigned> {
  static llvm::StringRef getName() { return "AtLeastNRegions"; }

  explicit AtLeastNRegions(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyAtLeastNRegions, num) {}
};
struct NResults : public BindArgTrait<unsigned> {
  static llvm::StringRef getName() { return "NResults"; }

  explicit NResults(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyNResults, num) {}
};
struct AtLeastNResults : public BindArgTrait<unsigned> {
  static llvm::StringRef getName() { return "AtLeastNResults"; }

  explicit AtLeastNResults(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyAtLeastNResults, num) {}
};
struct NSuccessors : public BindArgTrait<unsigned> {
  static llvm::StringRef getName() { return "NSuccessors"; }

  explicit NSuccessors(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyNSuccessors, num) {}
};
struct AtLeastNSuccessors : public BindArgTrait<unsigned> {
  static llvm::StringRef getName() { return "AtLeastNSuccessors"; }

  explicit AtLeastNSuccessors(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyAtLeastNSuccessors, num) {}
};

/// TODO Some standard traits not rebound (complexity/API restrictions):
/// - HasParent<>
/// - SingleBlockImplicitTerminator
/// - AutomaticAllocationScope
/// - ConstantLike

} // end namespace dmc
