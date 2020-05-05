#include "dmc/Traits/SpecTraits.h"
#include "dmc/Spec/SpecTypes.h"

using namespace mlir;

namespace dmc {

namespace {

template <typename TypeRange, typename ValueRange>
LogicalResult checkVariadicValues(TypeRange tys, ValueRange vals) {
  /// Check that the variadic array size can be determined.
  /// Not a fool-proof check. Count the number of values which belong
  /// to a variadic array.
  unsigned numVariadicTypes = llvm::count_if(tys,
      [](Type ty) { return ty.isa<VariadicType>(); });
  unsigned numNonVariadic = llvm::size(tys) - numVariadicTypes;
  unsigned numVariadicVals = llvm::size(vals) - numNonVariadic;
  return success(numVariadicVals % numVariadicTypes == 0);
}

template <typename TypeRange>
LogicalResult checkVariadicSegments(
    Operation *op, TypeRange tys, DenseIntElementsAttr sizes, StringRef name) {
  // `sizes` is checked as a 1D vector
  if (llvm::size(sizes) != llvm::size(tys))
    return op->emitOpError("incorrect number of ") << name << " segments; "
        << "expected " << llvm::size(tys) << " but got " << llvm::size(sizes);
  auto szIt = std::begin(sizes);
  auto szBegin = szIt;
  for (auto tyIt = std::begin(tys), tyEnd = std::end(tys);
       tyIt != tyEnd; ++tyIt, ++szIt) {
    if ((*szIt).getZExtValue() != 1 && !tyIt->template isa<VariadicType>())
      return op->emitOpError() << name << " #" << std::distance(szBegin, szIt)
          << " is not variadic but has non-unity segment size "
          << (*szIt).getZExtValue();
  }
  return success();
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
  if (failed(Base::verifyTrait(op)))
    return failure();
  auto *typeTrait = DynamicOperation::of(op)->getTrait<TypeConstraintTrait>();
  return checkVariadicSegments(op, typeTrait->getOpType().getInputs(),
                               getSegmentSizesAttr(op), "operand");
}

DenseIntElementsAttr SizedOperandSegments
::getSegmentSizesAttr(Operation *op) const {
  return op->getAttrOfType<DenseIntElementsAttr>(
      Base::getOperandSegmentSizeAttr());
}

LogicalResult SizedResultSegments::verifyOp(Operation *op) const {
  if (failed(Base::verifyTrait(op)))
    return failure();
  auto *typeTrait = DynamicOperation::of(op)->getTrait<TypeConstraintTrait>();
  return checkVariadicSegments(op, typeTrait->getOpType().getResults(),
                               getSegmentSizesAttr(op), "result");
}

DenseIntElementsAttr SizedResultSegments
::getSegmentSizesAttr(Operation *op) const {
  return op->getAttrOfType<DenseIntElementsAttr>(
      Base::getResultSegmentSizeAttr());
}

} // end namespace dmc
