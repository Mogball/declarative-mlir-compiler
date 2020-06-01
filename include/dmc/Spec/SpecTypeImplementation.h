#pragma once

#include "SpecKinds.h"
#include "OpType.h"
#include "dmc/Dynamic/DynamicOperation.h"

#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/Builders.h>

namespace dmc {

namespace SpecTypes {
bool is(mlir::Type base);
mlir::LogicalResult delegateVerify(mlir::Type base, mlir::Type ty);
} // end namespace SpecTypes

/// A SpecType is used to define a TypeConstraint. Each SpecType
/// implements a TypeConstraint called on DynamicOperations during
/// trait and Op verification.
template <typename ConcreteType, unsigned SpecKind,
          typename StorageType = mlir::DefaultTypeStorage>
class SpecType
    : public mlir::Type::TypeBase<ConcreteType, mlir::Type, StorageType> {
public:
  static constexpr auto Kind = SpecKind;

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
  /// Parser for simple types.
  static mlir::Type parse(mlir::DialectAsmParser &parser) {
    return get(parser.getBuilder().getContext());
  }
  /// Printer for simple types.
  void print(mlir::DialectAsmPrinter &printer) {
    printer << ConcreteType::getTypeName();
  }
};

/// Verify Type constraints.
namespace impl {
mlir::LogicalResult verifyTypeConstraints(mlir::Operation *op, OpType opTy);
} // end namespace impl

} // end namespace dmc
