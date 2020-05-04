#include "dmc/Traits/SpecTraits.h"

using namespace mlir;

namespace dmc {

namespace {

template <typename TypeRange, typename ValueRange>
LogicalResult checkVariadicValues(TypeRange tys, ValueRange vals) {
  /// Check that the variadic array size can be determined.
  /// Not a fool-proof check. Count the number of values which belong
  /// to a variadic array.
  unsigned numVariadicTypes = llvm::count_if(tys,
      [](Type ty) { return ty.isa<VariadicType>(); })
  unsigned numNonVariadic = llvm::size(tys) - numVariadicTypes;
  unsigned numVariadicVals = llvm::size(vals) - numNonVariadic;
  return success(numVariadicVals % numVariadic == 0);
}

} // end anonymous namespace

LogicalResult SameVariadicOperandSizes::verifyOp(Operation *op) const {
  auto *typeTrait = DynamicOperation::of(op)->getTrait<TypeConstraintTrait>();
  assert(typeTrait && "DynamicOperation missing TypeTrait");
  if (failed(checkVariadicValues(typeTrait->getOpType().getInputs(),
                                 op->getOperands())))
    return op->emitOpError("malformed variadic operands");
  return success();
}

LogicalResult SameVariadicResultSizes::verifyOp(Operation *op) const {
  auto *typeTrait = DynamicOperation::of(op)->getTrait<TypeConstraintTrait>();
  assert(typeTrait && "DynamicOperation missing TypeTrait");
  if (failed(checkVariadicValues(typeTrait->getOpType().getResults(),
                                 op->getResults())))
    return op->emitOpError("malformed variadic results");
  return success();
}

LogicalResult SizedOperandSegments::verifyOp(Operation *op) const {
}

LogicalResult SizedResultSegments::verifyOp(Operation *op) const {

}

} // end namespace dmc
