#pragma once

#include "SpecTypeImplementation.h"

#include <mlir/IR/StandardTypes.h>

namespace dmc {

/// SpecDialect types aim to capture type matching and verification logic.
/// E.g. !dmc.Int will verify the concrete type with ty.is<IntegerType>() and
/// !dmc.AnyOf<$Types...> will assert that the concrete type matches one of
/// the specified allowed types.
///
/// Variadic operands or results are specified with !dmc.Variadic<$Type>.
/// The following restrictions apply:
/// - Non-variadic values must preceed all variadic values
/// - If there are any variadic compound types, then the Op must have a
///   variadic size specifier OpTrait:
///
///     !dmc.Variadic<i32>, !dmc.Variadic<f32> is OK as the types
///     are mutually exclusive, but
///
///     !dmc.Variadic<!dmc.Float>, !dmc.Variadic<f32> requires a variadic 
///     size specifier (SameVariadicSize) as the types may not be 
///     mutually exclusive.
///
/// Optional values are specified with !dmc.Optional<$Type>. Optional values
/// must follow non-optional values and cannot be mixed with variadic values.
///
/// Custom type predicates can be specified with a call to a higher-level DSL,
/// e.g. a Python predicate.
namespace detail {
struct TypeListStorage;
struct WidthStorage;
struct WidthListStorage;
} // end namespace detail

namespace SpecTypes {
enum Kinds {
  Any = mlir::Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
  None,
  AnyOf,

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
  BF16
};
} // end namespace SpecTypes

/// Match any type.
class AnyType : public SimpleType<AnyType, SpecTypes::Any> {
public:
  using Base::Base; 
};

/// Match NoneType.
class NoneType : public SimpleType<NoneType, SpecTypes::None> {
public:
  using Base::Base;
  inline mlir::LogicalResult verify(mlir::Type ty) const override {
    return mlir::success(ty.isa<mlir::NoneType>());
  }
};

/// Match any Type in a list. The type list cannot be empty.
class AnyOfType 
    : public mlir::Type::TypeBase<AnyOfType, mlir::Type, 
                                  detail::TypeListStorage>,
      public SpecType {
public:
  using Base::Base;

  static inline bool kindof(unsigned kind) { 
    return kind == SpecTypes::AnyOf; 
  }

  static AnyOfType get(llvm::ArrayRef<mlir::Type> tys);
  static AnyOfType getChecked(mlir::Location loc, 
                              llvm::ArrayRef<mlir::Type> tys);

  /// Type list cannot be empty.
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, llvm::ArrayRef<mlir::Type> tys);
  /// Check Type is in the list.
  mlir::LogicalResult verify(mlir::Type ty) const override;
};

/// Match any IntegerType.
class AnyIntegerType 
    : public SimpleType<AnyIntegerType, SpecTypes::AnyInteger> {
public:
  using Base::Base;
  inline mlir::LogicalResult verify(mlir::Type ty) const override {
    return mlir::success(ty.isa<mlir::IntegerType>());
  }
};

/// Match any IntegerType of specified width.
class AnyIType
    : public mlir::Type::TypeBase<AnyIType, mlir::Type,
                                  detail::WidthStorage>,
      public SpecType {
public:
  using Base::Base;

  static inline bool kindof(unsigned kind) {
    return kind == SpecTypes::AnyI;
  }

  static AnyIType get(mlir::MLIRContext *ctx, unsigned width);
  static AnyIType getChecked(mlir::Location loc, unsigned width);

  /// Width must be one of [1, 8, 16, 32, 64].
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, unsigned width);
  /// Check Type is an integer of specified width.
  mlir::LogicalResult verify(mlir::Type ty) const override;
};

/// Match any IntegerType of the specified widths.
class AnyIntOfWidthsType
    : public mlir::Type::TypeBase<AnyIntOfWidthsType, mlir::Type,
                                  detail::WidthListStorage>,
      public SpecType {
public:
  using Base::Base;

  static inline bool kindof(unsigned kind) {
    return kind == SpecTypes::AnyIntOfWidths;
  }

  static AnyIntOfWidthsType get(mlir::MLIRContext *ctx, 
                                llvm::ArrayRef<unsigned> widths);
  static AnyIntOfWidthsType getChecked(mlir::Location loc,
                                       llvm::ArrayRef<unsigned> widths);

  /// Each width must be one of [1, 8, 16, 32, 64];
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, llvm::ArrayRef<unsigned> widths);
  /// Check Type is an integer of one of the specified widths.
  mlir::LogicalResult verify(mlir::Type ty) const override;
};

/// Match any signless integer.
/// TODO repeated similar code; codegen this or make shorthands.
class AnySignlessIntegerType : public SimpleType<
    AnySignlessIntegerType, SpecTypes::AnySignlessInteger> {
public:
  using Base::Base;
  inline mlir::LogicalResult verify(Type ty) const override {
    return mlir::success(ty.isSignlessInteger());
  }
};

/// Match a signless integer of a specified width.
class IType : public mlir::Type::TypeBase<IType, mlir::Type,
                                          detail::WidthStorage>,
              public SpecType {
public:
  using Base::Base;

  static inline bool kindof(unsigned kind) {
    return kind == SpecTypes::I;
  }

  static IType get(mlir::MLIRContext *ctx, unsigned width);
  static IType getChecked(mlir::Location loc, unsigned width);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, unsigned width);
  mlir::LogicalResult verify(mlir::Type ty) const override;
};

/// Match a signless integer of one of the specified widths.
class SignlessIntOfWidthsType 
    : public mlir::Type::TypeBase<SignlessIntOfWidthsType, mlir::Type,
                                  detail::WidthListStorage>,
      public SpecType {
public:
  using Base::Base;

  static inline bool kindof(unsigned kind) {
    return kind == SpecTypes::SignlessIntOfWidths;
  }

  static SignlessIntOfWidthsType get(
      mlir::MLIRContext *ctx, llvm::ArrayRef<unsigned> widths);
  static SignlessIntOfWidthsType getChecked(
      mlir::Location loc, llvm::ArrayRef<unsigned> widths);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, llvm::ArrayRef<unsigned> widths);
  mlir::LogicalResult verify(mlir::Type ty) const override;
};

/// Match any signed integer.
class AnySignedIntegerType 
    : public SimpleType<AnySignedIntegerType,
                        SpecTypes::AnySignedInteger> {
public:
  using Base::Base;
  inline mlir::LogicalResult verify(mlir::Type ty) const override {
    return mlir::success(ty.isSignedInteger());
  }
};

/// Match any signed integer of the specified width;
class SIType : public mlir::Type::TypeBase<SIType, mlir::Type,
                                           detail::WidthStorage>,
               public SpecType {
public:
  using Base::Base;

  static inline bool kindof(unsigned kind) {
    return kind == SpecTypes::SI;
  }

  static SIType get(mlir::MLIRContext *ctx, unsigned width);
  static SIType getChecked(mlir::Location loc, unsigned width);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, unsigned width);
  mlir::LogicalResult verify(Type ty) const override;
};

/// Match any signed integer of the specified widths.
class SignedIntOfWidthsType 
    : public mlir::Type::TypeBase<SignedIntOfWidthsType, mlir::Type,
                                  detail::WidthListStorage>,
      public SpecType {
public:
  using Base::Base;

  static inline bool kindof(unsigned kind) {
    return kind == SpecTypes::SignedIntOfWidths;
  }

  static SignedIntOfWidthsType get(
      mlir::MLIRContext *ctx, llvm::ArrayRef<unsigned> widths);
  static SignedIntOfWidthsType getChecked(
      mlir::Location, llvm::ArrayRef<unsigned> widths);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location, llvm::ArrayRef<unsigned> widths);
  mlir::LogicalResult verify(Type ty) const override;
};

/// Match any unsigned integer.
class AnyUnsignedIntegerType 
    : public SimpleType<AnyUnsignedIntegerType, 
                        SpecTypes::AnyUnsignedInteger> {
public:
  using Base::Base;
  inline mlir::LogicalResult verify(mlir::Type ty) const override {
    return mlir::success(ty.isUnsignedInteger());
  }
};

/// Match any unsigned integer of the specified width.
class UIType 
    : public mlir::Type::TypeBase<UIType, mlir::Type,
                                  detail::WidthStorage>,
      public SpecType {
public:
  using Base::Base;

  static inline bool kindof(unsigned kind) {
    return kind == SpecTypes::UI;
  }

  static UIType get(mlir::MLIRContext *ctx, unsigned width);
  static UIType getChecked(mlir::Location loc, unsigned width);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, unsigned width);
  mlir::LogicalResult verify(mlir::Type ty) const override;
};

class UnsignedIntOfWidthsType 
    : public mlir::Type::TypeBase<UnsignedIntOfWidthsType, mlir::Type,
                                  detail::WidthListStorage>,
      public SpecType {
public:
  using Base::Base;

  static inline bool kindof(unsigned kind) {
    return kind == SpecTypes::UnsignedIntOfWidths;
  }

  static UnsignedIntOfWidthsType get(
      mlir::MLIRContext *ctx, llvm::ArrayRef<unsigned> widths);
  static UnsignedIntOfWidthsType getChecked(
      mlir::Location loc, llvm::ArrayRef<unsigned> widths);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, llvm::ArrayRef<unsigned> widths);
  mlir::LogicalResult verify(mlir::Type ty) const override;
};

} // end namespace dmc
