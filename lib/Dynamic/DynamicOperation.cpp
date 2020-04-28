#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/Dynamic/DynamicOperation.h"

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>

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
    // TODO perform dynamic trait validation
    return success();
  }
};

/// Define base properies of all dynamic ops.
class BaseOp : public Op<BaseOp, DynamicOpTrait> {
public:
  using Op::Op;

  static void printAssembly(Operation *op, OpAsmPrinter &p) {
    // Assume we have the correct Op
    p.printGenericOp(op);
  }

  static LogicalResult verifyInvariants(Operation *op) {
    // TODO add call to custom verify() function
    // A DynamicOperation will always only have this trait
    return DynamicOpTrait::verifyTrait(op);
  }

  static LogicalResult foldHook(Operation *op, ArrayRef<Attribute> operands,
                                SmallVectorImpl<OpFoldResult> &results) {
    // TODO custom fold hooks
    return failure();
  }
};

AbstractOperation getBaseOpInfo(StringRef name, DynamicDialect *dialect,
                                TypeID typeId) {
  // TODO Custom functions, operand/result type/count checking
  // Add the dialect name
  return AbstractOperation(
      name, *dialect, BaseOp::getOperationProperties(),
      typeId, BaseOp::parseAssembly, BaseOp::printAssembly,
      BaseOp::verifyInvariants, BaseOp::foldHook, BaseOp::getCanonicalizationPatterns,
      BaseOp::getRawInterface, BaseOp::hasTrait);
}

} // end anonymous namespace

DynamicOperation::DynamicOperation(StringRef name, DynamicDialect *dialect)
    : DynamicObject{dialect->getDynContext()},
      name{(dialect->getNamespace() + "." + name).str()},
      opInfo{getBaseOpInfo(this->name, dialect, getTypeID())} {}

} // end namespace dmc
