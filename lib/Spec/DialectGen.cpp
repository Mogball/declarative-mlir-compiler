#include "dmc/Spec/DialectGen.h"
#include "dmc/Spec/SpecOps.h"
#include "dmc/Traits/StandardTraits.h"
#include "dmc/Traits/SpecTraits.h"

using namespace mlir;

namespace dmc {

void registerTrait(SymbolRefAttr traitSym, DynamicOperation *op) {
  /// TODO trait registry and lookup, which will also be used to register
  /// new dynamic traits.

}

void registerOp(OperationOp opOp, DynamicDialect *dialect) {
  /// Create the dynamic op.
  auto *op = dialect->createDynamicOp(op.getName());

  /// Add traits for fundamental properties.
  using std::make_unique; // TODO addOpTrait<TraitT>(args...)
  if (opOp.isTerminator())
    op->addOpTrait(make_unique<IsTerminator>());
  if (opOp.isCommutative())
    op->addOpTrait(make_unique<IsCommutative>());
  if (opOp.isIsolatedFromAbove())
    op->addOpTrait(make_unique<IsIsolatedFromAbove>());

  /// Add type and attribute constraint traits.
  op->addOpTrait(make_unique<TypeConstraintTrait>(opOp.getType()));
  op->addOpTrait(make_unique<AttrConstraintTrait>(opOp.getOpAttrs()));

  /// Process the remaining traits.
  for (auto traitSym : opOp.getOpTraits().getAsRange<SymbolRefAttr>())
    registerTrait(traitSym, op);
}

void registerDialect(DiaectOp dialectOp, DynamicContext *ctx) {
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
