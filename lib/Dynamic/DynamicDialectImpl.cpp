#include "dmc/Dynamic/Alias.h"
#include "dmc/Dynamic/DynamicDialect.h"
#include "dmc/Dynamic/DynamicType.h"
#include "dmc/Dynamic/DynamicAttribute.h"
#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/Dynamic/DynamicOperation.h"

#include <mlir/IR/Operation.h>
#include <llvm/ADT/StringMap.h>

using namespace mlir;
using namespace llvm;

namespace dmc {

class DynamicDialect::Impl {
  friend class DynamicDialect;

  StringMap<std::unique_ptr<DynamicOperation>> dynOps;
  StringMap<std::unique_ptr<DynamicTypeImpl>> dynTys;
  StringMap<std::unique_ptr<DynamicAttributeImpl>> dynAttrs;
  StringMap<TypeAlias> typeAliases;
  StringMap<AttributeAlias> attrAliases;

  /// Since an alias refers to a concrete value which may not be a dynamic
  /// value, metadata cannot be directly associated with the values. When
  /// registering aliases, map the aliased values back to the metadata.
  ///
  /// TODO This is not an ideal solution.
  llvm::DenseMap<Type, TypeAlias *> typeAliasData;
  llvm::DenseMap<Attribute, AttributeAlias *> attrAliasData;
};

DynamicDialect::~DynamicDialect() = default;

DynamicDialect::DynamicDialect(StringRef name, DynamicContext *ctx)
    : Dialect{name, ctx->getContext()},
      DynamicObject{ctx},
      impl{std::make_unique<Impl>()} {}

LogicalResult
DynamicDialect::registerDynamicOp(std::unique_ptr<DynamicOperation> op) {
  auto [it, inserted] = impl->dynOps.try_emplace(op->getOpInfo()->name,
                                                 std::move(op));
  return success(inserted);
}

DynamicOperation *DynamicDialect::lookupOp(StringRef name) const {
  auto it = impl->dynOps.find(name);
  return it == std::end(impl->dynOps) ? nullptr : it->second.get();
}

LogicalResult
DynamicDialect::registerDynamicType(std::unique_ptr<DynamicTypeImpl> type) {
  auto [it, inserted] = impl->dynTys.try_emplace(type->getName(),
                                                 std::move(type));
  return success(inserted);
}

DynamicTypeImpl *DynamicDialect::lookupType(StringRef name) const {
  auto it = impl->dynTys.find(name);
  return it == std::end(impl->dynTys) ? nullptr : it->second.get();
}

LogicalResult DynamicDialect
::registerDynamicAttr(std::unique_ptr<DynamicAttributeImpl> attr) {
  auto [it, inserted] = impl->dynAttrs.try_emplace(attr->getName(),
                                                   std::move(attr));
  return success(inserted);
}

DynamicAttributeImpl *DynamicDialect::lookupAttr(StringRef name) const {
  auto it = impl->dynAttrs.find(name);
  return it == std::end(impl->dynAttrs) ? nullptr : it->second.get();
}

LogicalResult DynamicDialect::registerTypeAlias(TypeAlias typeAlias) {
  auto [it, inserted] = impl->typeAliases.try_emplace(typeAlias.getName(),
                                                      typeAlias);
  return success(inserted);
}

TypeAlias *DynamicDialect::lookupTypeAlias(StringRef name) const {
  auto it = impl->typeAliases.find(name);
  return it == std::end(impl->typeAliases) ? nullptr : &it->second;
}

LogicalResult DynamicDialect::registerAttrAlias(AttributeAlias attrAlias) {
  auto [it, inserted] = impl->attrAliases.try_emplace(attrAlias.getName(),
                                                      attrAlias);
  return success(inserted);
}

AttributeAlias *DynamicDialect::lookupAttrAlias(StringRef name) const {
  auto it = impl->attrAliases.find(name);
  return it == std::end(impl->attrAliases) ? nullptr : &it->second;
}

TypeMetadata *DynamicDialect::lookupTypeData(mlir::Type type) {
  // Metadata directly associated with dynamic types
  if (auto dynTy = type.dyn_cast<DynamicType>())
    return dynTy.getTypeImpl();
  // Try to back-lookup a type alias
  if (auto it = impl->typeAliasData.find(type);
      it != std::end(impl->typeAliasData)) {
    return it->second;
  }
  return nullptr;
}

AttributeMetadata *DynamicDialect::lookupAttributeData(mlir::Attribute attr) {
  // Metadata directly associated with dynamic attributes
  if (auto dynAttr = attr.dyn_cast<DynamicAttribute>())
    return dynAttr.getAttrImpl();
  // Try to back-lookup an attribute alias
  if (auto it = impl->attrAliasData.find(attr);
      it != std::end(impl->attrAliasData)) {
    return it->second;
  }
  return nullptr;
}

} // end namespace dmc
