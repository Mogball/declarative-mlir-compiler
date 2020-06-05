#include "dmc/Dynamic/Alias.h"
#include "dmc/Dynamic/DynamicDialect.h"
#include "dmc/Spec/DialectGen.h"
#include "dmc/Spec/SpecOps.h"
#include "dmc/Spec/SpecTypes.h"
#include "dmc/Traits/Registry.h"
#include "dmc/Traits/StandardTraits.h"
#include "dmc/Traits/SpecTraits.h"
#include "dmc/Embed/OpFormatGen.h"
#include "dmc/Embed/InMemoryDef.h"

using namespace mlir;
using namespace mlir::dmc;

namespace dmc {

template <typename TypeRange>
unsigned countNonVariadicValues(TypeRange tys) {
  return llvm::count_if(tys, [](Type ty) { return !ty.isa<VariadicType>(); });
}

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

  /// Process user-defined traits.
  auto *registry = dialect->getContext()
      ->getRegisteredDialect<TraitRegistry>();
  for (auto trait : opOp.getOpTraits().getValue()) {
    auto ctor = registry->lookupTrait(trait.getName());
    if (failed(ctor.verify(opOp.getLoc(), trait.getParameters())))
      return failure();
    if (failed(op->addOpTrait(trait.getName(),
                              ctor.call(trait.getParameters()))))
      return opOp.emitOpError("duplicate op trait: ") << trait.getName();
  }

  /// Add region counts.
  auto opRegions = opOp.getOpRegions();
  auto numVarRegions = llvm::count_if(opRegions, [](Attribute opRegion) {
                                      return opRegion.isa<VariadicRegion>(); });
  if (!numVarRegions) {
    if (failed(op->addOpTrait<NRegions>(llvm::size(opRegions))))
      return opOp.emitOpError("specified an invalid op trait: ")
          << NRegions::getName();
  } else {
    if (!op->getTrait<AtLeastNRegions>())
      op->addOpTrait<AtLeastNRegions>(llvm::size(opRegions) - numVarRegions);
  }

  /// Add successor counts.
  auto opSuccs = opOp.getOpSuccessors();
  auto numVarSuccs = llvm::count_if(opSuccs, [](Attribute opSucc) {
                                    return opSucc.isa<VariadicSuccessor>(); });
  if (!numVarSuccs) {
    if (failed(op->addOpTrait<NSuccessors>(llvm::size(opSuccs))))
      return opOp.emitOpError("specified and invalid op trait: ")
          << NSuccessors::getName();
  } else {
    if (!op->getTrait<AtLeastNSuccessors>())
      op->addOpTrait<AtLeastNSuccessors>(llvm::size(opSuccs) - numVarSuccs);
  }

  /// Add operand and result count op traits if none exist. For variadic
  /// values, apply an AtLeast op trait.
  auto opTy = opOp.getOpType();
  if (!hasVariadicValues(opTy.getOperandTypes())) {
    if (failed(op->addOpTrait<NOperands>(opTy.getNumOperands())))
      return opOp.emitOpError("specified an invalid op trait: ")
          << NOperands::getName();
  } else {
    if (!op->getTrait<AtLeastNOperands>())
      op->addOpTrait<AtLeastNOperands>(countNonVariadicValues(
          opTy.getResultTypes()));
  }
  if (!hasVariadicValues(opTy.getResultTypes())) {
    if (failed(op->addOpTrait<NResults>(opTy.getNumResults())))
      return opOp.emitOpError("specified an invalid op trait: ")
          << NResults::getName();
  } else {
    if (!op->getTrait<AtLeastNResults>())
      op->addOpTrait<AtLeastNResults>(countNonVariadicValues(
          opTy.getResultTypes()));
  }

  /// Add type and attribute constraint traits last. Type and regions constrints
  /// depend on count traits to be checked beforehand.
  op->addOpTrait<TypeConstraintTrait>(opTy);
  op->addOpTrait<AttrConstraintTrait>(opOp.getOpAttrs());
  op->addOpTrait<RegionConstraintTrait>(opRegions);
  op->addOpTrait<SuccessorConstraintTrait>(opSuccs);

  /// Generate a custom op format, if one is specified.
  if (opOp.getAssemblyFormat()) {
    auto prefix = (dialect->getNamespace() + "__" + opOp.getName()).str();
    auto parserName = "parse__" + prefix;
    auto printerName = "print__" + prefix;
    py::InMemoryDef parser{parserName, "(parser, result)"};
    py::InMemoryDef printer{printerName, "(p, op)"};
    if (failed(generateOpFormat(opOp, parser.stream(), printer.stream())))
      return opOp.emitOpError("failed to generate op format");
    op->setOpFormat(std::move(parserName), std::move(printerName));
  }

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
    if (failed(dialect->registerTypeAlias({aliasOp.getName(), type,
                                          aliasOp.getBuilder()})))
      return aliasOp.emitOpError("a type alias with this name already exists");
  } else {
    if (dialect->lookupAttr(aliasOp.getName()))
      return aliasOp.emitOpError("an attribute with this name already exists");
    if (failed(dialect->registerAttrAlias({
        aliasOp.getName(), aliasOp.getAliasedAttr(), aliasOp.getBuilder(),
        aliasOp.getAttrType()})))
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
