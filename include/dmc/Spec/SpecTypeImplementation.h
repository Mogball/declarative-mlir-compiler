#pragma once

#include <mlir/IR/Types.h>

namespace dmc {

/// A SpecType is used to define a TypeConstraint. Each SpecType 
/// implements a TypeConstraint called on DynamicOperations during
/// trait and Op verification.
template <typename ConcreteType, unsigned Kind, 
          typename StorageType = mlir::DefaultTypeStorage>
class SpecType 
    : public mlir::Type::TypeBase<ConcreteType, mlir::Type, StorageType> {
public:
  /// Explicitly define Base class for templated classes.
  using Parent = mlir::Type::TypeBase<ConcreteType, mlir::Type, StorageType>;
  using Base = SpecType<ConcreteType, Kind, StorageType>;

  /// Inherit parent constructors to pass onto child classes.
  using Parent::Parent;

  /// All SpecType subclasses implement a function of the signature
  ///
  /// LogicalResult verify(Type ty)
  /// 
  /// Which executes the TypeConstraint. Because mlir::Type is a CRTP
  /// class, we have manually create a virtual table using the kind.

  /// Compare type kinds.
  static bool kindof(unsigned kind) { return kind == Kind; }
};

/// Simple type shorthand class.
template <typename ConcreteType, unsigned Kind>
class SimpleType : public SpecType<ConcreteType, Kind> {
public:
  using Parent = SpecType<ConcreteType, Kind>;
  using Base = SimpleType<ConcreteType, Kind>;

  using Parent::Parent;

  /// Dispatch to simple Type getter.
  static ConcreteType get(mlir::MLIRContext *ctx) {
    return Parent::get(ctx, Kind);
  }
};

} // end namespace dmc
