#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/Dynamic/DynamicOperation.h"
#include "dmc/Dynamic/DynamicDialect.h"
#include "dmc/Embed/ParserPrinter.h"
#include "dmc/Spec/SpecAttrs.h"
#include "dmc/Traits/SpecTraits.h"

#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Builders.h>

using namespace mlir;

namespace dmc {

ParseResult BaseOp::parseAssembly(OpAsmParser &parser,
                                 OperationState &result) {
  // TODO This is a *three*-step lookup
  auto *ctx = parser.getBuilder().getContext();
  auto *dynCtx = ctx->getRegisteredDialect<DynamicContext>();
  assert(dynCtx && "Dynamic context is not registered");
  auto *dialect = dynCtx->lookupDialectFor(result.name);
  assert(dialect && "Not a dynamic operation");
  auto *dynOp = dialect->lookupOp(result.name);
  return dynOp->parseOperation(parser, result);
}

DynamicOperation *DynamicOperation::of(Operation *op) {
  /// All DynamicOperations must belong to a DynamicDialect.
  auto *dialect = dynamic_cast<DynamicDialect *>(op->getDialect());
  assert(dialect && "Dynamic operation belongs to a non-dynamic dialect?");
  return dialect->lookupOp(op->getName());
}

bool BaseOp::classof(mlir::Operation *op) {
  return dynamic_cast<DynamicDialect *>(op->getDialect());
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

void DynamicOperation::setOpFormat(std::string parserName,
                                   std::string printerName) {
  parserFcn = std::move(parserName);
  printerFcn = std::move(printerName);
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

AbstractOperation::OperationProperties
DynamicOperation::getOpProperties() const {
  AbstractOperation::OperationProperties props{};
  for (const auto &trait : traits) {
    props |= trait.second->getTraitProperties();
  }
  return props;
}

ParseResult DynamicOperation::parseOperation(OpAsmParser &parser,
                                             OperationState &result) {
  if (parserFcn) {
    if (!py::execParser(*parserFcn, parser, result))
      return failure();
  } else {
    return parser.emitError(parser.getCurrentLocation(),
                            "op does not have a custom parser");
  }

  // Handle default-valued attributes
  auto opAttrs = getTrait<AttrConstraintTrait>()->getOpAttrs();
  DenseSet<Identifier> found;
  for (auto &[name, attr] : result.attributes)
    found.insert(name);
  for (auto &[name, attr] : opAttrs) {
    if (auto def = attr.dyn_cast<DefaultAttr>(); def && !found.count(name)) {
      result.attributes.push_back({name, def.getDefaultValue()});
    }
  }
  return success();
}

void DynamicOperation::printOperation(OpAsmPrinter &printer, Operation *op) {
  if (printerFcn) {
    py::execPrinter(*printerFcn, printer, op, this);
  } else {
    printer.printGenericOp(op);
  }
}

} // end namespace dmc
