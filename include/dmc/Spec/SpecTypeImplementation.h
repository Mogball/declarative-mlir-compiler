#pragma once

#include "dmc/Kind.h"
#include "dmc/Dynamic/DynamicOperation.h"

namespace dmc {

namespace SpecTypes {
enum Kinds {
  Any = Kind::FIRST_SPEC_TYPE,
  None,
  AnyOf,
  AllOf,

  AnyInteger,
  AnyI,
  AnyIntOfWidths,

  AnySignlessInteger,
  I,
  SignlessIntOfWidths,

  AnySignedInteger,
  SI,
  SignedIntOfWidths,

  AnyUnsignedInteger,
  UI,
  UnsignedIntOfWidths,

  Index,

  AnyFloat,
  F,
  FloatOfWidths,
  BF16,

  AnyComplex,
  Complex,

  Opaque,
  Function,

  Variadic, // Optional is a subset of Variadic

  Isa,

  NUM_TYPES
};

bool is(mlir::Type base);
mlir::LogicalResult delegateVerify(mlir::Type base, mlir::Type ty);

} // end namespace SpecTypes

/// A SpecType is used to define a TypeConstraint. Each SpecType
/// implements a TypeConstraint called on DynamicOperations during
/// trait and Op verification.
template <typename ConcreteType, unsigned Kind,
          typename StorageType = mlir::DefaultTypeStorage>
class SpecType
    : public mlir::Type::TypeBase<ConcreteType, mlir::Type, StorageType> {
  friend class SpecDialect;

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

  /// Get the type storage.
  auto getImpl() { return Parent::getImpl(); }
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

/// Verify Type constraints.
namespace impl {
mlir::LogicalResult verifyTypeConstraints(
    mlir::Operation *op, mlir::FunctionType opTy);
} // end namespace impl

} // end namespace dmc
