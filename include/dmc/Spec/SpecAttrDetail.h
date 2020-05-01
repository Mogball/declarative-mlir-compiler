#pragma once

#include "SpecAttrImplementation.h"

#include <mlir/IR/Types.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Diagnostics.h>

namespace dmc {

/// Place full declaration in header to allow template usage.
namespace detail {

struct TypedAttrStorage : public mlir::AttributeStorage {
  using KeyTy = mlir::Type;

  explicit TypedAttrStorage(KeyTy key);
  bool operator==(const KeyTy &key) const;
  static llvm::hash_code hashKey(const KeyTy &key);
  static TypedAttrStorage *construct(
      mlir::AttributeStorageAllocator &alloc, const KeyTy &key);

  KeyTy type;
};

} // end namespace detail

/// AttrConstraint on an IntegerAttr with a specified underlying Type.
template <unsigned Kind, typename AttrT, typename UnderlyingT>
class TypedAttrBase 
    : public SpecAttr<TypedAttrBase<Kind, AttrT, UnderlyingT>, 
                      Kind, detail::TypedAttrStorage> {
public:
  using Base = TypedAttrBase<Kind, AttrT, UnderlyingT>;
  using Parent = SpecAttr<Base, Kind, detail::TypedAttrStorage>;
  using Parent::Parent;

  static inline TypedAttrBase get(mlir::Type intType) {
    return Parent::get(intType.getContext(), Kind, intType);
  }

  static inline TypedAttrBase getChecked(
      mlir::Location loc, mlir::Type intType) {
    return Parent::getChecked(loc, Kind, intType); 
  }

  static inline mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, mlir::Type intType);

  inline mlir::LogicalResult verify(mlir::Attribute attr) {
    return mlir::success(attr.isa<AttrT>() &&
        mlir::succeeded(this->getImpl()->type.template cast<UnderlyingT>()
            .verify(attr.cast<AttrT>().getType())));
  }
};

/// Out-of-line definitions
template <unsigned Kind, typename AttrT, typename UnderlyingT>
mlir::LogicalResult TypedAttrBase<Kind, AttrT, UnderlyingT>
::verifyConstructionInvariants(mlir::Location loc, mlir::Type intType) {
  if (!intType)
    return mlir::emitError(loc) << "Type cannot be null";
  if (!intType.isa<UnderlyingT>())
    return mlir::emitError(loc) << "Invalid underlying type: " << intType;
  return mlir::success();
}

} // end namespace dmc
