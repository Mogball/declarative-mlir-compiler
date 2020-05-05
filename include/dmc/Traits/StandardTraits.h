#pragma once

#include "Kinds.h"
#include "dmc/Dynamic/DynamicOperation.h"

/// Bind common OpTraits into DynamicTraits
namespace dmc {

template <template <typename ConcreteType> class BaseTrait, unsigned Kind>
struct BindTrait : public DynamicTrait<Kind> {
  mlir::LogicalResult verifyOp(mlir::Operation *op) const override {
    // Provide dummy template arg
    return BaseTrait<DynamicTrait>::verifyTrait(op);
  }

  mlir::AbstractOperation::OperationProperties
  getTraitProperties() const override {
    return BaseTrait<DynamicTrait>::getTraitProperties();
  }
};
template <typename Arg, unsigned Kind>
class BindArgTrait : public DynamicTrait<Kind> {
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
    : public BindTrait<mlir::OpTrait::IsTerminator,
                       Traits::IsTerminator> {};
struct IsCommutative
    : public BindTrait<mlir::OpTrait::IsCommutative,
                       Traits::IsCommutative> {};
struct IsIsolatedFromAbove
    : public BindTrait<mlir::OpTrait::IsIsolatedFromAbove,
                       Traits::IsIsolatedFromAbove> {};

struct OperandsAreFloatLike
    : public BindTrait<mlir::OpTrait::OperandsAreFloatLike,
                       Traits::OperandsAreFloatLike> {};
struct OperandsAreSignlessIntegerLike
    : public BindTrait<mlir::OpTrait::OperandsAreSignlessIntegerLike,
                       Traits::OperandsAreSignlessIntegerLike> {};
struct ResultsAreBoolLike
    : public BindTrait<mlir::OpTrait::ResultsAreBoolLike,
                       Traits::ResultsAreBoolLike> {};
struct ResultsAreFloatLike
    : public BindTrait<mlir::OpTrait::ResultsAreFloatLike,
                       Traits::ResultsAreFloatLike> {};
struct ResultsAreSignlessIntegerLike
    : public BindTrait<mlir::OpTrait::ResultsAreSignlessIntegerLike,
                       Traits::ResultsAreSignlessIntegerLike> {};

struct SameOperandsShape
    : public BindTrait<mlir::OpTrait::SameOperandsShape,
                       Traits::SameOperandsShape> {};
struct SameOperandsAndResultShape
    : public BindTrait<mlir::OpTrait::SameOperandsAndResultShape,
                       Traits::SameOperandsAndResultShape> {};
struct SameOperandsElementType
    : public BindTrait<mlir::OpTrait::SameOperandsElementType,
                       Traits::SameOperandsElementType> {};
struct SameOperandsAndResultElementType
    : public BindTrait<mlir::OpTrait::SameOperandsAndResultElementType,
                       Traits::SameOperandsAndResultElementType> {};
struct SameOperandsAndResultType
    : public BindTrait<mlir::OpTrait::SameOperandsAndResultType,
                       Traits::SameOperandsAndResultType> {};
struct SameTypeOperands
    : public BindTrait<mlir::OpTrait::SameTypeOperands,
                       Traits::SameTypeOperands> {};

/// Stateful traits require constructors.
struct NOperands
    : public BindArgTrait<unsigned, Traits::NOperands> {
  explicit NOperands(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyNOperands, num) {}
};
struct AtLeastNOperands
    : public BindArgTrait<unsigned, Traits::AtLeastNOperands> {
  explicit AtLeastNOperands(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyAtLeastNOperands, num) {}
};
struct NRegions
    : public BindArgTrait<unsigned, Traits::NRegions> {
  explicit NRegions(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyNRegions, num) {}
};
struct AtLeastNRegions
    : public BindArgTrait<unsigned, Traits::AtLeastNRegions> {
  explicit AtLeastNRegions(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyAtLeastNRegions, num) {}
};
struct NResults
    : public BindArgTrait<unsigned, Traits::NResults> {
  explicit NResults(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyNResults, num) {}
};
struct AtLeastNResults
    : public BindArgTrait<unsigned, Traits::AtLeastNResults> {
  explicit AtLeastNResults(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyAtLeastNResults, num) {}
};
struct NSuccessors
    : public BindArgTrait<unsigned, Traits::NSuccessors> {
  explicit NSuccessors(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyNSuccessors, num) {}
};
struct AtLeastNSuccessors
    : public BindArgTrait<unsigned, Traits::AtLeastNSuccessors> {
  explicit AtLeastNSuccessors(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyAtLeastNSuccessors, num) {}
};

/// TODO Some standard traits not rebound (complexity/API restrictions):
/// - HasParent<>
/// - SingleBlockImplicitTerminator
/// - AutomaticAllocationScope
/// - ConstantLike

} // end namespace dmc
