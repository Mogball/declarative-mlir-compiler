#pragma once

#include "Kinds.h"
#include "dmc/Dynamic/DynamicOperation.h"
#include "dmc/Spec/SpecTypeImplementation.h"
#include "dmc/Spec/SpecAttrImplementation.h"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/OpDefinition.h>

namespace dmc {

/// For operands and results, more than one variadic value requires a size
/// specification trait. Only one of SameVariadicSizes or SizedSegments may
/// be used.
///
/// These traits mark an Operation with variadic operands or results with a
/// size specification that all variadic values have the same array size.
struct SameVariadicOperandSizes
    : public DynamicTrait<Traits::SameVariadicOperandSizes> {
  /// Verify variadic operand formation.
  mlir::LogicalResult verifyOp(mlir::Operation *op) const override;

  inline static mlir::SymbolRefAttr getSymbol(mlir::MLIRContext *ctx) {
    return mlir::SymbolRefAttr::get("SameVariadicOperandSizes", ctx);
  }
};
struct SameVariadicResultSizes
    : public DynamicTrait<Traits::SameVariadicResultSizes> {
  /// Verify variadic result formation.
  mlir::LogicalResult verifyOp(mlir::Operation *op) const override;

  inline static mlir::SymbolRefAttr getSymbol(mlir::MLIRContext *ctx) {
    return mlir::SymbolRefAttr::get("SameVariadicResultSizes", ctx);
  }
};

/// These traits indicate that the Operation has variadic operands or result
/// with sizes known at runtime, captured inside an attribute.
struct SizedOperandSegments
    : public DynamicTrait<Traits::SizedOperandSegments> {
  /// Based off mlir::AttrSizedOperandSegments.
  using Base = mlir::OpTrait::AttrSizedOperandSegments<DynamicTrait>;

  /// Verify the operand segment size attribute.
  mlir::LogicalResult verifyOp(mlir::Operation *op) const override;

  /// Get the operand segment size attribute.
  mlir::DenseIntElementsAttr getSegmentSizesAttr(mlir::Operation *op) const;

  inline static mlir::SymbolRefAttr getSymbol(mlir::MLIRContext *ctx) {
    return mlir::SymbolRefAttr::get("SizedOperandSegments", ctx);
  }
};
struct SizedResultSegments
    : public DynamicTrait<Traits::SizedResultSegments> {
  /// Based off mlir::AttrSizedResultSegments.
  using Base = mlir::OpTrait::AttrSizedResultSegments<DynamicTrait>;

  /// Verify the result segment size attribute.
  mlir::LogicalResult verifyOp(mlir::Operation *op) const override;

  /// Get the result segment size attribute.
  mlir::DenseIntElementsAttr getSegmentSizesAttr(mlir::Operation *op) const;

  inline static mlir::SymbolRefAttr getSymbol(mlir::MLIRContext *ctx) {
    return mlir::SymbolRefAttr::get("SizedResultSegments", ctx);
  }
};

/// Top-level Type and Attribute verifiers apply the specified constraints
/// on an Operation. Trait validity and interactions are already verified on
/// the OperationOp spec.
class TypeConstraintTrait : public DynamicTrait<Traits::TypeConstraintTrait> {
public:
  /// Create a type constraint with types wrapped in a FunctionType.
  inline explicit TypeConstraintTrait(mlir::FunctionType opTy)
      : opTy{opTy} {}

  /// Check the Op's operand and result types.
  inline mlir::LogicalResult verifyOp(mlir::Operation *op) const override {
    return impl::verifyTypeConstraints(op, opTy);
  }

  inline auto getOpType() { return opTy; }

private:
  mlir::FunctionType opTy;
};


class AttrConstraintTrait : public DynamicTrait<Traits::AttrConstraintTrait> {
public:
  /// Create an attribute constraint with attributes
  /// wrapped in a DictionaryAttr.
  inline explicit AttrConstraintTrait(mlir::DictionaryAttr opAttrs)
      : opAttrs{opAttrs} {}

  /// Check the Op's attributes.
  inline mlir::LogicalResult verifyOp(mlir::Operation *op) const override {
    return impl::verifyAttrConstraints(op, opAttrs);
  }

  inline auto getOpAttrs() { return opAttrs; }

private:
  mlir::DictionaryAttr opAttrs;
};

} // end namespace dmc
