#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Operation.h>

namespace dmc {

namespace SpecAttrs {
enum Kinds {
  Any = mlir::Attribute::Kind::FIRST_PRIVATE_EXPERIMENTAL_0_ATTR,
  Bool,
  Index,
  APInt,

  AnyI,
  I,
  SI,
  UI,
  F,

  String,
  Type,
  Unit,
  Dictionary,
  Elements,
  Array,

  SymbolRef,
  FlatSymbolRef,

  Constant,
  AnyOf,
  AllOf,
  OfType,

  NUM_ATTRS
};

bool is(mlir::Attribute base);
mlir::LogicalResult delegateVerify(mlir::Attribute base,
                                   mlir::Attribute attr);

} // end namespace SpecAttrs

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

/// Verify Attribute constraints.
namespace impl {
mlir::LogicalResult verifyAttrConstraints(
    mlir::Operation *op, mlir::DictionaryAttr opAttrs);
} // end namespace impl

} // end namespace dmc
