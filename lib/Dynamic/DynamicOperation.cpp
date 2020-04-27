#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/Dynamic/DynamicOperation.h"

#include <mlir/IR/OpDefinition.h>

using namespace mlir;

namespace dmc {
namespace {

/// Mark dynamic operations with this OpTrait. Also, Op requires
/// at least one OpTrait.
template <typename ConcreteType>
class DynamicOpTrait : 
    public OpTrait::TraitBase<ConcreteType, DynamicOpTrait> {
public:
  static LogicalResult verifyTrait(Operation *) {
    // TODO perform some dynamic op validation?
    return success();
  }
};

/// Define base properies of all dynamic ops.
class BaseOp : public Op<BaseOp, DynamicOpTrait> {
public:
  using Op::Op;

  static StringRef getOperationName() { return "base_op"; }
};

AbstractOperation getBaseOpInfo(StringRef name, DynamicDialect *dialect, 
                                TypeID typeId) {
  // TODO Custom functions, operand/result type/count checking
  // Add the dialect name
  auto opName = dialect->getNamespace() + "." + name;
  return AbstractOperation(
      opName.str(), *dialect, BaseOp::getOperationProperties(),
      typeId, BaseOp::parseAssembly, BaseOp::printAssembly,
      BaseOp::verifyInvariants, BaseOp::foldHook, BaseOp::getCanonicalizationPatterns,
      BaseOp::getRawInterface, BaseOp::hasTrait);
}

} // end anonymous namespace

DynamicOperation::DynamicOperation(StringRef name, DynamicDialect *dialect)
    : DynamicObject{dialect->getDynContext()},
      opInfo{getBaseOpInfo(name, dialect, getTypeID())} {}

} // end namespace dmc
