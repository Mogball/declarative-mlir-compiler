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
  static LogicalResult verifyTrait(Operation *op) {
    // Hook into the DynamicTraits
    return DynamicOperation::of(op)->verifyOpTraits(op);
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

} // end anonymous namespace

DynamicOperation *DynamicOperation::of(Operation *op) {
  /// All DynamicOperations must belong to a DynamicDialect.
  /// TODO mlir::Dialect should indicate if it is dynamic.
  auto *dialect = static_cast<DynamicDialect *>(op->getDialect());
  return dialect->getDynContext()->lookupOp(op);
}

DynamicOperation::DynamicOperation(StringRef name, DynamicDialect *dialect)
    : DynamicObject{dialect->getDynContext()},
      name{(dialect->getNamespace() + "." + name).str()},
      dialect{dialect} {}

void DynamicOperation::addOpTrait(
    StringRef name, std::unique_ptr<DynamicTrait> trait) {
  auto it = traits.try_emplace(name, std::move(trait)); (void) it;
  assert(it.second && "Trait already exists");
}

void DynamicOperation::finalize() {
  // Add the operation to the dialect
  dialect->addOperation({
      name, *dialect, getOpProperties(), getTypeID(),
      BaseOp::parseAssembly, BaseOp::printAssembly,
      BaseOp::verifyInvariants, BaseOp::foldHook,
      BaseOp::getCanonicalizationPatterns,
      BaseOp::getRawInterface, BaseOp::hasTrait
  });
  // Grab the reference to the allocated AbstractOperation
  opInfo = AbstractOperation::lookup(
      name, dialect->getContext());
  assert(opInfo != nullptr && "Failed to add DynamicOperation");
  // Give ownership to DynamicContext
  getDynContext()->registerDynamicOp(this);
}

LogicalResult DynamicOperation::verifyOpTraits(Operation *op) const {
  for (const auto &trait : traits) {
    if (failed(trait.second->verifyOp(op))) {
      return failure();
    }
  }
  return success();
}

AbstractOperation::OperationProperties DynamicOperation::getOpProperties() const {
  AbstractOperation::OperationProperties props{};
  for (const auto &trait : traits) {
    props |= trait.second->getTraitProperties();
  }
  return props;
}

} // end namespace dmc
