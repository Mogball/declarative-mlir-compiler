#include "dmc/Spec/SpecTypeSwitch.h"
#include "dmc/Traits/SpecTraits.h"

#include <mlir/IR/Operation.h>

using namespace mlir;

namespace dmc {
namespace SpecTypes {

bool is(Type base) {
  return Any <= base.getKind() && base.getKind() < NUM_TYPES;
}

struct VerifyAction {
  Type argTy; // type to verify

  template <typename ConcreteType>
  LogicalResult operator()(ConcreteType base) const {
    return base.verify(argTy);
  }
};

/// Big switch table.
LogicalResult delegateVerify(Type base, Type ty) {
  VerifyAction action{ty};
  return SpecTypes::kindSwitch(action, base);
}

} // end namespace SpecTypes

/// Type verification.
namespace impl {

template <typename TypeRange>
LogicalResult verifyTypeRange(Operation *op, ArrayRef<Type> baseTys,
                              TypeRange tys, StringRef name) {
  auto firstTy = std::begin(tys), lastTy = std::end(tys);
  auto tyIt = firstTy;
  for (auto baseIt = std::begin(baseTys), baseEnd = std::end(baseTys);
       baseIt != baseEnd || tyIt != lastTy; ++tyIt, ++baseIt) {
    if (baseIt == baseEnd)
      return op->emitOpError("too many ") << name << "s";
    if (tyIt == lastTy)
      return op->emitOpError("not enough ") << name << "s";
    if ((SpecTypes::is(*baseIt) &&
         failed(SpecTypes::delegateVerify(*baseIt, *tyIt))) ||
        (!SpecTypes::is(*baseIt) && *baseIt != *tyIt))
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
    if (tyIt == tyEnd)
      return op->emitOpError("too many ") << name << 's';
    if (std::begin(values) == std::end(values) && !tyIt->isa<VariadicType>())
      return op->emitOpError("not enough ") << name << 's';
    for (auto valIt = std::begin(values), valEnd = std::end(values);
         valIt != valEnd; ++valIt, ++valIdx) {
      // TODO custom type descriptions with dynamic types
      auto valType = (*valIt).getType();
      if ((SpecTypes::is(*tyIt) &&
           failed(SpecTypes::delegateVerify(*tyIt, valType))) ||
          (!SpecTypes::is(*tyIt) && *tyIt != valType))
        return op->emitOpError() << name << " #" << valIdx << " must be "
            << *tyIt << " but got " << valType;
    }
  }
  return success();
}

LogicalResult verifyOperandTypes(Operation *op, mlir::FunctionType opTy,
                                 DynamicOperation *info) {
  if (info->getTrait<SizedOperandSegments>()) {
    return verifyVariadicTypes(op, opTy.getInputs(),
        SizedOperandSegments::getOperandGroup, "operand");
  } else if (info->getTrait<SameVariadicOperandSizes>()) {
    return verifyVariadicTypes(op, opTy.getInputs(),
        SameVariadicOperandSizes::getOperandGroup, "operand");
  } else {
    return verifyTypeRange(op, opTy.getInputs(),
        op->getOperandTypes(), "operand");
  }
}

LogicalResult verifyResultTypes(Operation *op, mlir::FunctionType opTy,
                                DynamicOperation *info) {
  if (info->getTrait<SizedResultSegments>()) {
    return verifyVariadicTypes(op, opTy.getResults(),
        SizedResultSegments::getResultGroup, "result");
  } else if (info->getTrait<SameVariadicResultSizes>()) {
    return verifyVariadicTypes(op, opTy.getResults(),
        SameVariadicResultSizes::getResultGroup, "result");
  } else {
    return verifyTypeRange(op, opTy.getResults(),
        op->getResultTypes(), "result");
  }
}

LogicalResult verifyTypeConstraints(Operation *op, mlir::FunctionType opTy) {
  auto *info = DynamicOperation::of(op);
  return failure(failed(verifyOperandTypes(op, opTy, info)) ||
                 failed(verifyResultTypes(op, opTy, info)));
}

} // end namespace impl

} // end namespace dmc
