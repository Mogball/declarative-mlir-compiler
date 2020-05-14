#include "dmc/Dynamic/DynamicDialect.h"
#include "dmc/Spec/DialectGen.h"
#include "dmc/Spec/SpecOps.h"
#include "dmc/Spec/SpecTypes.h"
#include "dmc/Spec/LowerOpaqueType.h"
#include "dmc/Traits/Registry.h"
#include "dmc/Traits/StandardTraits.h"
#include "dmc/Traits/SpecTraits.h"

using namespace mlir;

namespace dmc {

LogicalResult registerOp(OperationOp opOp, DynamicDialect *dialect) {
  /// Create the dynamic op.
  auto op = dialect->createDynamicOp(opOp.getName().getValue());

  /// Add traits for fundamental properties.
  if (opOp.isTerminator())
    op->addOpTrait<IsTerminator>();
  if (opOp.isCommutative())
    op->addOpTrait<IsCommutative>();
  if (opOp.isIsolatedFromAbove())
    op->addOpTrait<IsIsolatedFromAbove>();

  /// Add number of operands, results, successors, and regions traits.
  /// TODO support for successors and regions.
  op->addOpTrait<NRegions>(0);
  op->addOpTrait<NSuccessors>(0);
  // Add operand/result counts if none are variadic.
  if (!hasVariadicValues(opOp.getType().getInputs()))
    op->addOpTrait<NOperands>(llvm::size(opOp.getType().getInputs()));
  if (!hasVariadicValues(opOp.getType().getResults()))
    op->addOpTrait<NResults>(llvm::size(opOp.getType().getResults()));

  /// Process the remaining traits.
  auto *registry = dialect->getContext()
      ->getRegisteredDialect<TraitRegistry>();
  for (auto traitSym : opOp.getOpTraits().getAsRange<FlatSymbolRefAttr>()) {
    auto traitName = traitSym.getValue();
    op->addOpTrait(traitName, registry->lookupTrait(traitName));
  }

  /// Add type and attribute constraint traits last.
  op->addOpTrait<TypeConstraintTrait>(opOp.getType());
  op->addOpTrait<AttrConstraintTrait>(opOp.getOpAttrs());

  /// Finally, register the Op.
  if (failed(op->finalize()) ||
      failed(dialect->registerDynamicOp(std::move(op))))
    return opOp.emitOpError("an operation with this name already exists");
  return success();
}

LogicalResult registerType(TypeOp typeOp, DynamicDialect *dialect) {
  if (failed(dialect->createDynamicType(
        typeOp.getName(), typeOp.getParameters())))
    return typeOp.emitOpError("a type with this name already exists");
  return success();
}

LogicalResult registerDialect(DialectOp dialectOp, DynamicContext *ctx) {
  /// Create the dynamic dialect
  auto *dialect = ctx->createDynamicDialect(dialectOp.getName());
  dialect->allowUnknownOperations(dialectOp.allowsUnknownOps());
  dialect->allowUnknownTypes(dialectOp.allowsUnknownTypes());

  /// First create the Types
  for (auto typeOp : dialectOp.getOps<TypeOp>()) {
    if (failed(registerType(typeOp, dialect)))
      return failure();
  }

  /// Rewrite operations with newly registered types.
  if (failed(lowerOpaqueTypes(dialectOp)))
    return failure();

  /// Create the Ops
  for (auto opOp : dialectOp.getOps<OperationOp>()) {
    if (failed(registerOp(opOp, dialect)))
      return failure();
  }

  return success();
}

LogicalResult registerAllDialects(ModuleOp dialects, DynamicContext *ctx) {
  for (auto dialectOp : dialects.getOps<DialectOp>()) {
    if (failed(registerDialect(dialectOp, ctx)))
      return failure();
  }
  return success();
}

} // end namespace dmc
