#pragma once

#include <mlir/IR/Attributes.h>

namespace dmc {

template <typename ConcreteType, unsigned Kind,
          typename StorageType = mlir::AttributeStorage>
class SpecAttr
    : public mlir::Attribute::AttrBase<ConcreteType, mlir::Attribute, 
                                                     StorageType> {
  friend class SpecDialect;

public:
  using Parent = mlir::Attribute::AttrBase<ConcreteType, mlir::Attribute,
                                           StorageType>;
  using Base = SpecAttr<ConcreteType, Kind, StorageType>;
  using Parent::Parent;

  static bool kindof(unsigned kind) { return kind == Kind; }

  auto getImpl() { return Parent::getImpl(); }
};

template <typename ConcreteType, unsigned Kind>
class SimpleAttr : public SpecAttr<ConcreteType, Kind> {
public:
  using Parent = SpecAttr<ConcreteType, Kind>;
  using Base = SimpleAttr<ConcreteType, Kind>;
  using Parent::Parent;

  static ConcreteType get(mlir::MLIRContext *ctx) {
    return Parent::get(ctx, Kind);
  }
};

} // end namespace dmc
