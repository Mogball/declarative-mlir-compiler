#include "dmc/Dynamic/Alias.h"
#include "dmc/Dynamic/DynamicDialect.h"
#include "dmc/Spec/DialectGen.h"
#include "dmc/Spec/SpecOps.h"
#include "dmc/Spec/SpecTypes.h"
#include "dmc/Traits/Registry.h"
#include "dmc/Traits/StandardTraits.h"
#include "dmc/Traits/SpecTraits.h"

using namespace mlir;
using namespace mlir::dmc;

namespace dmc {

LogicalResult registerOp(OperationOp opOp, DynamicDialect *dialect) {
  /// Create the dynamic op.
  auto op = dialect->createDynamicOp(opOp.getName());

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
  for (auto traitSym : opOp.getOpTraits().getValue()) {
    if (!traitSym.getParameters().empty())
      return opOp.emitOpError("parameterized op traits currently unsupported");
    auto traitName = traitSym.getName();
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

LogicalResult registerAttr(AttributeOp attrOp, DynamicDialect *dialect) {
  if (failed(dialect->createDynamicAttr(
        attrOp.getName(), attrOp.getParameters())))
    return attrOp.emitOpError("an attribute with this name already exists");
  return success();
}

LogicalResult registerAlias(AliasOp aliasOp, DynamicDialect *dialect) {
  /// Need to check that a type with the alias name does not already exist.
  if (auto type = aliasOp.getAliasedType()) {
    if (dialect->lookupType(aliasOp.getName()))
      return aliasOp.emitOpError("a type with this name already exists");
    if (failed(dialect->registerTypeAlias({aliasOp.getName(), type})))
      return aliasOp.emitOpError("a type alias with this name already exists");
  } else {
    if (dialect->lookupAttr(aliasOp.getName()))
      return aliasOp.emitOpError("an attribute with this name already exists");
    if (failed(dialect->registerAttrAlias({aliasOp.getName(),
                                          aliasOp.getAliasedAttr()})))
      return aliasOp.emitOpError(
          "an attribute alias with  this name already exists");
  }
  return success();
}

LogicalResult registerDialect(DialectOp dialectOp, DynamicContext *ctx) {
  /// Create the dynamic dialect
  auto *dialect = ctx->createDynamicDialect(dialectOp.getName());
  dialect->allowUnknownOperations(dialectOp.allowsUnknownOps());
  dialect->allowUnknownTypes(dialectOp.allowsUnknownTypes());

  /// Walk the children operations.
  for (auto &specOp : dialectOp) {
    /// If the op can be reparsed, do so.
    if (auto reparseOp = dyn_cast<ReparseOpInterface>(&specOp))
      if (failed(reparseOp.reparse()))
        return failure();
    /// Op-specific actions.
    if (auto opOp = dyn_cast<OperationOp>(&specOp)) {
      if (failed(registerOp(opOp, dialect)))
        return failure();
    } else if (auto typeOp = dyn_cast<TypeOp>(&specOp)) {
      if (failed(registerType(typeOp, dialect)))
        return failure();
    } else if (auto attrOp = dyn_cast<AttributeOp>(&specOp)) {
      if (failed(registerAttr(attrOp, dialect)))
        return failure();
    } else if (auto aliasOp = dyn_cast<AliasOp>(&specOp)) {
      if (failed(registerAlias(aliasOp, dialect)))
        return failure();
    }
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
