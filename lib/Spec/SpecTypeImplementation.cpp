#include "dmc/Spec/SpecTypeSwitch.h"
#include "dmc/Traits/SpecTraits.h"

#include <mlir/IR/Operation.h>

using namespace mlir;

namespace dmc {
namespace SpecTypes {

bool is(Type base) {
  return Any <= base.getKind() && base.getKind() < LAST_SPEC_TYPE;
}

LogicalResult delegateVerify(Type base, Type ty) {
  /// If not a type constraint, use a direct comparison.
  if (!is(base))
    return success(base == ty);
  /// Use the switch table.
  VerifyAction<Type> action{ty};
  return SpecTypes::kindSwitch(action, base);
}

} // end namespace SpecTypes

/// Type verification.
namespace impl {

template <typename TypeRange>
LogicalResult verifyTypeRange(Operation *op, ArrayRef<Type> baseTys,
                              TypeRange tys, StringRef name) {
  auto firstTy = std::begin(tys), tyEnd = std::end(tys);
  auto tyIt = firstTy;
  for (auto baseIt = std::begin(baseTys), baseEnd = std::end(baseTys);
       baseIt != baseEnd || tyIt != tyEnd; ++tyIt, ++baseIt) {
    /// Number of operands and results are verified by previous traits.
    assert(baseIt != baseEnd && tyIt != tyEnd);
    if (failed(SpecTypes::delegateVerify(*baseIt, *tyIt)))
      return op->emitOpError() << name << " #" << std::distance(firstTy, tyIt)
          << " must be " << *baseIt << " but got " << *tyIt;
  }
  return success();
}

template <typename GetValueGroup>
LogicalResult verifyVariadicTypes(Operation *op, ArrayRef<Type> baseTys,
                                  GetValueGroup getValues, StringRef name) {
  unsigned groupIdx = 0, valIdx = 0;
  auto values = getValues(op, groupIdx);
  for (auto tyIt = std::begin(baseTys), tyEnd = std::end(baseTys);
       tyIt != tyEnd || std::begin(values) != std::end(values);
       ++tyIt, values = getValues(op, ++groupIdx)) {
    assert(tyIt != tyEnd);
    assert(std::begin(values) != std::end(values) || tyIt->isa<VariadicType>());
    for (auto valIt = std::begin(values), valEnd = std::end(values);
         valIt != valEnd; ++valIt, ++valIdx) {
      // TODO custom type descriptions with dynamic types
      auto valType = (*valIt).getType();
      if (failed(SpecTypes::delegateVerify(*tyIt, valType)))
        return op->emitOpError() << name << " #" << valIdx << " must be "
            << *tyIt << " but got " << valType;
    }
  }
  return success();
}

LogicalResult
verifyOperandTypes(Operation *op, OpType opTy, DynamicOperation *info) {
  if (info->getTrait<SizedOperandSegments>()) {
    return verifyVariadicTypes(op, opTy.getOperandTypes(),
        SizedOperandSegments::getOperandGroup, "operand");
  } else if (info->getTrait<SameVariadicOperandSizes>()) {
    return verifyVariadicTypes(op, opTy.getOperandTypes(),
        SameVariadicOperandSizes::getOperandGroup, "operand");
  } else {
    return verifyTypeRange(op, opTy.getOperandTypes(),
        op->getOperandTypes(), "operand");
  }
}

LogicalResult
verifyResultTypes(Operation *op, OpType opTy, DynamicOperation *info) {
  if (info->getTrait<SizedResultSegments>()) {
    return verifyVariadicTypes(op, opTy.getResultTypes(),
        SizedResultSegments::getResultGroup, "result");
  } else if (info->getTrait<SameVariadicResultSizes>()) {
    return verifyVariadicTypes(op, opTy.getResultTypes(),
        SameVariadicResultSizes::getResultGroup, "result");
  } else {
    return verifyTypeRange(op, opTy.getResultTypes(),
        op->getResultTypes(), "result");
  }
}

LogicalResult verifyTypeConstraints(Operation *op, OpType opTy) {
  auto *info = DynamicOperation::of(op);
  return failure(failed(verifyOperandTypes(op, opTy, info)) ||
                 failed(verifyResultTypes(op, opTy, info)));
}

} // end namespace impl

} // end namespace dmc
