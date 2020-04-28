#pragma once

#include "dmc/Dynamic/DynamicOperation.h"

/// Bind common OpTraits into DynamicTraits
namespace dmc {

template <template <typename ConcreteType> class BaseTrait>
struct BindTrait : public DynamicTrait {
  mlir::LogicalResult verifyOp(mlir::Operation *op) const override {
    // Provide dummy template arg
    return BaseTrait<int>::verifyTrait(op);
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
struct SameOperandsShape          : public BindTrait<mlir::OpTrait::SameOperandsShape>          {};
struct SameOperandsAndResultShape : public BindTrait<mlir::OpTrait::SameOperandsAndResultShape> {};
struct SameOperandsElementType    : public BindTrait<mlir::OpTrait::SameOperandsElementType>    {};
struct SameOperandsAndResultElementType : public BindTrait<mlir::OpTrait::SameOperandsAndResultElementType> {};
struct SameOperandsAndResultType  : public BindTrait<mlir::OpTrait::SameOperandsAndResultType>  {};
struct SameTypeOperands           : public BindTrait<mlir::OpTrait::SameTypeOperands>           {};

struct IsTerminator  : public BindTrait<mlir::OpTrait::IsTerminator>  {};
struct IsIsolatedFromAbove : public BindTrait<mlir::OpTrait::IsIsolatedFromAbove> {};
// IsCommutative is an Op property but not a verifiable trait

struct OperandsAreFloatLike : public BindTrait<mlir::OpTrait::OperandsAreFloatLike> {};
struct OperandsAreSignlessIntegerLike : public BindTrait<mlir::OpTrait::OperandsAreSignlessIntegerLike> {};
struct ResultsAreBoolLike   : public BindTrait<mlir::OpTrait::ResultsAreBoolLike>   {};
struct ResultsAreFloatLike  : public BindTrait<mlir::OpTrait::ResultsAreFloatLike>  {};
struct ResultsAreSignlessIntegerLike : public BindTrait<mlir::OpTrait::ResultsAreSignlessIntegerLike> {};

struct ZeroOperands  : public BindTrait<mlir::OpTrait::ZeroOperands>  {};
struct OneOperand    : public BindTrait<mlir::OpTrait::OneOperand>    {};
struct ZeroRegion    : public BindTrait<mlir::OpTrait::ZeroRegion>    {};
struct OneRegion     : public BindTrait<mlir::OpTrait::OneRegion>     {};
struct ZeroResult    : public BindTrait<mlir::OpTrait::ZeroResult>    {};
struct OneResult     : public BindTrait<mlir::OpTrait::OneResult>     {};
struct ZeroSuccessor : public BindTrait<mlir::OpTrait::ZeroSuccessor> {};
struct OneSuccessor  : public BindTrait<mlir::OpTrait::OneSuccessor>  {};

/// Stateful traits require constructors.
struct NOperands          : public BindArgTrait<unsigned> {
  explicit NOperands(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyNOperands,          num) {}
};
struct AtLeastNOperands   : public BindArgTrait<unsigned> {
  explicit AtLeastNOperands(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyAtLeastNOperands,   num) {}
};
struct NRegions           : public BindArgTrait<unsigned> {
  explicit NRegions(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyNRegions,           num) {}
};
struct AtLeastNRegions    : public BindArgTrait<unsigned> {
  explicit AtLeastNRegions(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyAtLeastNRegions,    num) {}
};
struct NResults           : public BindArgTrait<unsigned> {
  explicit NResults(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyNResults,           num) {}
};
struct AtLeastNResults    : public BindArgTrait<unsigned> {
  explicit AtLeastNResults(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyAtLeastNResults,    num) {}
};
struct NSuccessors        : public BindArgTrait<unsigned> {
  explicit NSuccessors(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyNSuccessors,        num) {}
};
struct AtLeastNSuccessors : public BindArgTrait<unsigned> {
  explicit AtLeastNSuccessors(ArgTy num)
      : BindArgTrait(mlir::OpTrait::impl::verifyAtLeastNSuccessors, num) {}
};

/// TODO Some standard traits not rebound (complexity/API restrictions):
/// - AttrSizedResultSegments
/// - AttrSizedOperandSegments
/// - HasParent
/// - SingleBlockImplicitTerminator
/// - AutomaticAllocationScope
/// - ConstantLike

} // end namespace dmc
