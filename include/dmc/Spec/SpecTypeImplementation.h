#pragma once

#include <mlir/IR/Types.h>

namespace dmc {

/// A SpecType is used to define a TypeConstraint. Each SpecType 
/// implements a TypeConstraint called on DynamicOperations during
/// trait and Op verification.
class SpecType {
public:
  inline virtual ~SpecType() = default;

  inline virtual mlir::LogicalResult 
  verify(mlir::Type ty) const { return mlir::success(); }
};

/// Simple type shorthand class.
template <typename ConcreteType, unsigned Kind>
class SimpleType : public mlir::Type::TypeBase<ConcreteType, mlir::Type>, 
                   public SpecType {
public:
  using Parent = mlir::Type::TypeBase<ConcreteType, mlir::Type>;
  using Base = SimpleType<ConcreteType, Kind>;

  /// Compare against Kind.
  static bool kindof(unsigned kind) { return kind == Kind; }

  /// Dispatch to simple Type getter.
  static ConcreteType get(mlir::MLIRContext *ctx) {
    return Parent::get(ctx, Kind);
  }
};

} // end namespace dmc
