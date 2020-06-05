#include "dmc/Traits/SpecTraits.h"
#include "dmc/Spec/SpecTypes.h"

#include <numeric>

using namespace mlir;

namespace dmc {

namespace {

OpType getOpType(Operation *op) {
  auto *typeTrait = DynamicOperation::of(op)->getTrait<TypeConstraintTrait>();
  assert(typeTrait && "DynamicOperation missing TypeTrait");
  return typeTrait->getOpType();
}

template <typename TypeRange>
auto calcFixedVariadicSize(TypeRange tys, ValueRange vals) {
  unsigned numVariadicTypes = llvm::count_if(tys,
      [](Type ty) { return ty.isa<VariadicType>(); });
  unsigned numNonVariadic = llvm::size(tys) - numVariadicTypes;
  unsigned numVariadicVals = llvm::size(vals) - numNonVariadic;
  return std::make_tuple(numVariadicVals, numVariadicTypes);
}

template <typename TypeRange>
LogicalResult checkVariadicValues(TypeRange tys, ValueRange vals) {
  /// Check that the variadic array size can be determined.
  /// Not a fool-proof check. Count the number of values which belong
  /// to a variadic array.
  auto [numVariadicVals, numVariadicTypes] = calcFixedVariadicSize(tys, vals);
  return success(numVariadicVals % numVariadicTypes == 0);
}

template <typename TypeRange>
unsigned getFixedSegmentSize(TypeRange tys, ValueRange vals) {
  auto [numVariadicVals, numVariadicTypes] = calcFixedVariadicSize(tys, vals);
  return numVariadicVals / numVariadicTypes; // previous checked as divisible
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
    if ((*szIt).getZExtValue() != 1 && !(*tyIt).template isa<VariadicType>())
      return op->emitOpError() << name << " #" << std::distance(szBegin, szIt)
          << " is not variadic but has non-unity segment size "
          << (*szIt).getZExtValue();
  }
  return success();
}

} // end anonymous namespace

LogicalResult SameVariadicOperandSizes::verifyOp(Operation *op) const {
  if (failed(checkVariadicValues(getOpType(op).getOperandTypes(),
                                 op->getOperands())))
    return op->emitOpError("malformed variadic operands");
  return success();
}

LogicalResult SameVariadicResultSizes::verifyOp(Operation *op) const {
  if (failed(checkVariadicValues(getOpType(op).getResultTypes(),
                                 op->getResults())))
    return op->emitOpError("malformed variadic results");
  return success();
}

LogicalResult SizedOperandSegments::verifyOp(Operation *op) const {
  if (failed(Base::verifyTrait(op)))
    return failure();
  return checkVariadicSegments(op, getOpType(op).getOperandTypes(),
                               getSegmentSizesAttr(op), "operand");
}

DenseIntElementsAttr SizedOperandSegments::getSegmentSizesAttr(Operation *op) {
  return op->getAttrOfType<DenseIntElementsAttr>(
      Base::getOperandSegmentSizeAttr());
}

LogicalResult SizedResultSegments::verifyOp(Operation *op) const {
  if (failed(Base::verifyTrait(op)))
    return failure();
  return checkVariadicSegments(op, getOpType(op).getResultTypes(),
                               getSegmentSizesAttr(op), "result");
}

DenseIntElementsAttr SizedResultSegments::getSegmentSizesAttr(Operation *op) {
  return op->getAttrOfType<DenseIntElementsAttr>(
      Base::getResultSegmentSizeAttr());
}

template <typename ValueRange, typename TypeRange, typename GetSegSize>
ValueRange getValueGroup(TypeRange tys, ValueRange vals, unsigned idx,
                         GetSegSize getNextSegSize) {
  if (idx >= llvm::size(tys)) // out-of-bounds: return "null"
    return {std::end(vals), std::end(vals)};

  auto firstTy = std::begin(tys), lastTy = std::next(firstTy, idx);
  auto startIdx = std::accumulate(firstTy, lastTy, 0,
      [&](unsigned acc, Type ty) { return acc + getNextSegSize(ty); });
  auto firstVal = std::next(std::begin(vals), startIdx);
  return {firstVal, std::next(firstVal, getNextSegSize(*lastTy))};
}

template <typename ValueRange, typename TypeRange>
ValueRange getFixedValueGroup(TypeRange tys, ValueRange vals, unsigned idx) {
  auto segSize = getFixedSegmentSize(tys, vals);
  auto getNextSegSize = [segSize](Type ty) {
    return ty.isa<VariadicType>() ? segSize : 1;
  };
  return getValueGroup<ValueRange>(tys, vals, idx, getNextSegSize);
}

template <typename ValueRange, typename TypeRange>
ValueRange getAttrValueGroup(TypeRange tys, ValueRange vals, unsigned idx,
                             DenseIntElementsAttr sizes) {
  auto szIt = std::begin(sizes);
  auto getNextSegSize = [&szIt](Type) {
    auto cur = szIt++;
    return (*cur).getZExtValue();
  };
  return getValueGroup<ValueRange>(tys, vals, idx, getNextSegSize);
}

/// Value group getters for variadic values.
ValueRange SameVariadicOperandSizes::getGroup(
    Operation *op, unsigned idx) {
  return getFixedValueGroup<OperandRange>(getOpType(op).getOperandTypes(),
      op->getOperands(), idx);
}

ValueRange SameVariadicResultSizes::getGroup(
    Operation *op, unsigned idx) {
  return getFixedValueGroup<ResultRange>(getOpType(op).getResultTypes(),
      op->getResults(), idx);
}

ValueRange SizedOperandSegments::getGroup(
    Operation *op, unsigned idx) {
  return getAttrValueGroup<OperandRange>(getOpType(op).getOperandTypes(),
      op->getOperands(), idx, getSegmentSizesAttr(op));
}

ValueRange SizedResultSegments::getGroup(
    Operation *op, unsigned idx) {
  return getAttrValueGroup<ResultRange>(getOpType(op).getResultTypes(),
      op->getResults(), idx, getSegmentSizesAttr(op));
}

} // end namespace dmc
