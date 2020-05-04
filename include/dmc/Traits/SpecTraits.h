#pragma once

#include "dmc/Dynamic/DynamicOperation.h"
#include "dmc/Spec/SpecTypeImplementation.h"
#include "dmc/Spec/SpecAttrImplementation.h"

namespace dmc {

/// For operands and results, more than one variadic value requires a size
/// specification trait. Only one of SameVariadicSizes or SizedSegments may
/// be used.
///
/// These traits mark an Operation with variadic operands or results with a
/// size specification that all variadic values have the same array size.
struct SameVariadicOperandSizes : public DynamicTrait {
  mlir::LogicalResult verifyOp(mlir::Operation *op) const override;
};
struct SameVariadicResultSizes : public DynamicTrait {
  mlir::LogicalResult verifyOp(mlir::Operation *op) const override;
};

/// These traits indicate that the Operation has variadic operands or result
/// with sizes known at runtime, captured inside an attribute.
struct SizedOperandSegments : public DynamicTrait {
  /// Based off mlir::AttrSizedOperandSegments.
  mlir::LogicalResult verifyOp(mlir::Operation *op) const override;
};
struct SizedResultSegments : public DynamicTrait {
  /// Based off mlir::AttrSizedResultSegments.
  mlir::LogicalResult verifyOp(mlir::Operation *op) const override;
};

/// Top-level Type and Attribute verifiers apply the specified constraints
/// on an Operation. Trait validity and interactions are already verified on
/// the OperationOp spec.
class TypeConstraintTrait : public DynamicTrait {
public:
  inline explicit TypeConstraintTrait(mlir::FunctionType opTy)
      : opTy{opTy} {}

  inline mlir::LogicalResult verifyOp(mlir::Operation *op) const override {
    return impl::verifyTypeConstraints(op, opTy);
  }

  inline auto getOpType() { return opTy; }

private:
  mlir::FunctionType opTy;
};


class AttrConstraintTrait : public DynamicTrait {
public:
  inline explicit AttrConstraintTrait(mlir::DictionaryAttr opAttrs)
      : opAttrs{opAttrs} {}

  inline mlir::LogicalResult verifyOp(mlir::Operation *op) const override {
    return impl::verifyAttrConstraints(op, opAttrs);
  }

  inline auto getOpAttrs() { return opAttrs; }

private:
  mlir::DictionaryAttr opAttrs;
};

} // end namespace dmc
