#pragma once

#include "dmc/Dynamic/DynamicOperation.h"

/// For operands and results, more than one variadic value requires a size
/// specification trait. Only one of SameVariadicSizes or SizedSegments may
/// be used.
namespace dmc {

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
  /// Based of mlir::AttrSizedOperandSegments.
  mlir::LogicalResult verifyOp(mlir::Operation *op) const override;
};
struct SizedResultSegments : public DynamicTrait {
  /// Based of mlir::AttrSizedResultSegments.
  mlir::LogicalResult verifyOp(mlir::Operation *op) const override;
};

} // end namespace dmc
