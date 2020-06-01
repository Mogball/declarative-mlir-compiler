#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/Dynamic/DynamicOperation.h"
#include "dmc/Dynamic/DynamicDialect.h"

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
  auto *dialect = dynamic_cast<DynamicDialect *>(op->getDialect());
  assert(dialect && "Dynamic operation belongs to a non-dynamic dialect?");
  return dialect->lookupOp(op->getName().getStringRef());
}

DynamicOperation::DynamicOperation(StringRef name, DynamicDialect *dialect)
    : DynamicObject{dialect->getDynContext()},
      name{(dialect->getNamespace() + "." + name).str()},
      dialect{dialect} {}

LogicalResult DynamicOperation::addOpTrait(
    StringRef name, std::unique_ptr<DynamicTrait> trait) {
  if (getTrait(name))
    return failure();
  traits.emplace_back(name, std::move(trait));
  return success();
}

DynamicTrait *DynamicOperation::getTrait(StringRef name) {
  for (auto &trait : traits) {
    if (trait.first == name)
      return trait.second.get();
  }
  return nullptr;
}

LogicalResult DynamicOperation::finalize() {
  // Check that the operation name is unused.
  if (AbstractOperation::lookup(name, dialect->getContext()))
    return failure();
  // Add the operation to the dialect
  dialect->addOperation({
      name, *dialect, getOpProperties(), getTypeID(),
      BaseOp::parseAssembly, BaseOp::printAssembly,
      BaseOp::verifyInvariants, BaseOp::foldHook,
      BaseOp::getCanonicalizationPatterns,
      BaseOp::getRawInterface, BaseOp::hasTrait
  });
  /// Take reference to the operation info.
  opInfo = AbstractOperation::lookup(name, dialect->getContext());
  assert(opInfo != nullptr && "Failed to add DynamicOperation");
  return success();
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
