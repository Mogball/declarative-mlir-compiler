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

  llvm::DenseMap<OperationName, std::unique_ptr<DynamicOperation>> dynOps;
  StringMap<std::unique_ptr<DynamicTypeImpl>> dynTys;
  StringMap<std::unique_ptr<DynamicAttributeImpl>> dynAttrs;
  StringMap<TypeAlias> typeAliases;
  StringMap<AttributeAlias> attrAliases;

  /// Since an alias refers to a concrete value which may not be a dynamic
  /// value, metadata cannot be directly associated with the values. When
  /// registering aliases, map the aliased values back to the metadata.
  ///
  /// TODO This is not an ideal solution.
  llvm::DenseMap<Type, TypeAlias> typeAliasData;
  llvm::DenseMap<Attribute, AttributeAlias> attrAliasData;
};

DynamicDialect::~DynamicDialect() = default;

DynamicDialect::DynamicDialect(StringRef name, DynamicContext *ctx)
    : Dialect{name, ctx->getContext()},
      DynamicObject{ctx},
      impl{std::make_unique<Impl>()} {}

LogicalResult
DynamicDialect::registerDynamicOp(std::unique_ptr<DynamicOperation> op) {
  auto *opInfo = op->getOpInfo();
  if (auto [it, inserted] = impl->dynOps.try_emplace(
      op->getOpInfo(), std::move(op)); !inserted)
    return failure();
  return getDynContext()->registerDialectSymbol(this, opInfo);
}

DynamicOperation *DynamicDialect::lookupOp(OperationName name) const {
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
  if (auto [it, inserted] = impl->typeAliases.try_emplace(
      typeAlias.getName(), typeAlias); !inserted)
    return failure();
  if (auto [it, inserted] = impl->typeAliasData.try_emplace(
      typeAlias.getAliasedType(), typeAlias); !inserted)
    return failure();
  return getDynContext()->registerDialectSymbol(this,
                                                typeAlias.getAliasedType());
}

TypeAlias *DynamicDialect::lookupTypeAlias(StringRef name) const {
  auto it = impl->typeAliases.find(name);
  return it == std::end(impl->typeAliases) ? nullptr : &it->second;
}

LogicalResult DynamicDialect::registerAttrAlias(AttributeAlias attrAlias) {
  if (auto [it, inserted] = impl->attrAliases.try_emplace(
      attrAlias.getName(), attrAlias); !inserted)
    return failure();
  if (auto [it, inserted] = impl->attrAliasData.try_emplace(
      attrAlias.getAliasedAttr(), attrAlias); !inserted)
    return failure();
  return getDynContext()->registerDialectSymbol(this,
                                                attrAlias.getAliasedAttr());
}

AttributeAlias *DynamicDialect::lookupAttrAlias(StringRef name) const {
  auto it = impl->attrAliases.find(name);
  return it == std::end(impl->attrAliases) ? nullptr : &it->second;
}

TypeMetadata *DynamicDialect::lookupTypeData(mlir::Type type) {
  // Metadata directly associated with dynamic types
  if (auto dynTy = type.dyn_cast<DynamicType>())
    return dynTy.getDynImpl();
  // Try to back-lookup a type alias
  if (auto it = impl->typeAliasData.find(type);
      it != std::end(impl->typeAliasData)) {
    return &it->second;
  }
  return nullptr;
}

AttributeMetadata *DynamicDialect::lookupAttributeData(mlir::Attribute attr) {
  // Metadata directly associated with dynamic attributes
  if (auto dynAttr = attr.dyn_cast<DynamicAttribute>())
    return dynAttr.getDynImpl();
  // Try to back-lookup an attribute alias
  if (auto it = impl->attrAliasData.find(attr);
      it != std::end(impl->attrAliasData)) {
    return &it->second;
  }
  return nullptr;
}

template <typename E, typename MapT> static auto getDialectObjs(MapT &objs) {
  std::vector<E *> ret;
  ret.reserve(std::size(objs));
  for (auto &obj : llvm::make_second_range(objs))
    ret.push_back(obj.get());
  return ret;
}

std::vector<DynamicOperation *> DynamicDialect::getOps() {
  return getDialectObjs<DynamicOperation>(impl->dynOps);
}

std::vector<DynamicTypeImpl *> DynamicDialect::getTypes() {
  return getDialectObjs<DynamicTypeImpl>(impl->dynTys);
}

std::vector<DynamicAttributeImpl *> DynamicDialect::getAttributes() {
  return getDialectObjs<DynamicAttributeImpl>(impl->dynAttrs);
}

} // end namespace dmc
