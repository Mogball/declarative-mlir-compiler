#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/Dynamic/DynamicOperation.h"

#include <mlir/IR/OpDefinition.h>

using namespace mlir;

namespace dmc {
namespace {

class BaseOp : public Op<BaseOp> {
public:
  using Op::Op;

  static StringRef getOperationName() { return "base_op"; }
};

AbstractOperation getBaseOpInfo(StringRef name, DynamicDialect *dialect, TypeID typeId) {
  // TODO Custom functions
  return AbstractOperation(
      name, *dialect, BaseOp::getOperationProperties(),
      typeId, BaseOp::parseAssembly, BaseOp::printAssembly,
      BaseOp::verifyInvariants, BaseOp::foldHook, BaseOp::getCanonicalizationPatterns,
      BaseOp::getRawInterface, BaseOp::hasTrait);
}

} // end anonymous namespace

DynamicOperation::DynamicOperation(StringRef name, DynamicDialect *dialect)
    : DynamicObject{dialect->getDynContext()},
      opInfo{getBaseOpInfo(name, dialect, getTypeID())} {}

} // end namespace dmc
