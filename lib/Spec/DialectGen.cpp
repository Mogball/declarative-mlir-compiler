#include "dmc/Spec/DialectGen.h"
#include "dmc/Spec/SpecOps.h"
#include "dmc/Spec/SpecTypes.h"
#include "dmc/Traits/Registry.h"
#include "dmc/Traits/StandardTraits.h"
#include "dmc/Traits/SpecTraits.h"

using namespace mlir;

namespace dmc {

void registerOp(OperationOp opOp, DynamicDialect *dialect) {
  /// Create the dynamic op.
  auto *op = dialect->createDynamicOp(opOp.getName().getValue());

  /// Add traits for fundamental properties.
  if (opOp.isTerminator())
    op->addOpTrait<IsTerminator>();
  if (opOp.isCommutative())
    op->addOpTrait<IsCommutative>();
  if (opOp.isIsolatedFromAbove())
    op->addOpTrait<IsIsolatedFromAbove>();

  /// Add type and attribute constraint traits.
  op->addOpTrait<TypeConstraintTrait>(opOp.getType());
  op->addOpTrait<AttrConstraintTrait>(opOp.getOpAttrs());

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
}

void registerDialect(DialectOp dialectOp, DynamicContext *ctx) {
  /// Create the dynamic dialect
  auto *dialect = ctx->createDynamicDialect(dialectOp.getName());
  dialect->allowUnknownOperations(dialectOp.allowsUnknownOps());
  dialect->allowUnknownTypes(dialectOp.allowsUnknownTypes());

  /// Create the Ops
  for (auto opOp : dialectOp.getOps<OperationOp>())
    registerOp(opOp, dialect);
}

void registerAllDialects(ModuleOp dialects, DynamicContext *ctx) {
  for (auto dialectOp : dialects.getOps<DialectOp>())
    registerDialect(dialectOp, ctx);
}

} // end namespace dmc
