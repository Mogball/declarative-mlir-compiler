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
template <typename ConcreteType, unsigned Kind,
          typename AttrT, typename UnderlyingT>
class TypedAttrBase
    : public SpecAttr<ConcreteType, Kind, detail::TypedAttrStorage> {
public:
  using Base = TypedAttrBase<ConcreteType, Kind, AttrT, UnderlyingT>;
  using Parent = SpecAttr<ConcreteType, Kind, detail::TypedAttrStorage>;
  using Underlying = UnderlyingT;
  using Parent::Parent;

  static ConcreteType get(UnderlyingT ty) {
    return Parent::get(ty.getContext(), Kind, ty);
  }

  static ConcreteType getChecked(
      mlir::Location loc, UnderlyingT ty) {
    return Parent::getChecked(loc, Kind, ty);
  }

  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, UnderlyingT ty) {
    if (!ty)
      return mlir::emitError(loc) << "Type cannot be null";
    return mlir::success();
  }

  mlir::LogicalResult verify(mlir::Attribute attr) {
    return mlir::success(attr.isa<AttrT>() &&
        mlir::succeeded(this->getImpl()->type.template cast<UnderlyingT>()
            .verify(attr.cast<AttrT>().getType())));
  }
};

} // end namespace dmc
