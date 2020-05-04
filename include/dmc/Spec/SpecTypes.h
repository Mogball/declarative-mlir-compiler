#pragma once

#include "SpecTypeImplementation.h"

#include <mlir/IR/Dialect.h>
#include <mlir/IR/StandardTypes.h>

namespace dmc {

/// SpecDialect types aim to capture type matching and verification logic.
/// E.g. !dmc.Int will verify the concrete type with ty.is<IntegerType>() and
/// !dmc.AnyOf<$Types...> will assert that the concrete type matches one of
/// the specified allowed types.
///
/// Variadic operands or results are specified with !dmc.Variadic<$Type>.
/// More than one variadic operand requires a size specifier trait.
///
/// Custom type predicates can be specified with a call to a higher-level DSL,
/// e.g. a Python predicate.
namespace detail {
struct TypeListStorage;
struct WidthStorage;
struct WidthListStorage;
struct OneTypeStorage;
struct OpaqueTypeStorage;
} // end namespace detail

/// Match any type.
class AnyType : public SimpleType<AnyType, SpecTypes::Any> {
public:
  using Base::Base;
  inline mlir::LogicalResult verify(Type) { return mlir::success(); }
};

/// Match NoneType.
class NoneType : public SimpleType<NoneType, SpecTypes::None> {
public:
  using Base::Base;
  inline mlir::LogicalResult verify(Type ty) {
    return mlir::success(ty.isa<mlir::NoneType>());
  }
};

/// Match any Type in a list. The type list cannot be empty.
class AnyOfType : public SpecType<AnyOfType, SpecTypes::AnyOf,
                                  detail::TypeListStorage> {
public:
  using Base::Base;

  static AnyOfType get(llvm::ArrayRef<Type> tys);
  static AnyOfType getChecked(mlir::Location loc, llvm::ArrayRef<Type> tys);

  /// Type list cannot be empty.
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, llvm::ArrayRef<Type> tys);
  /// Check Type is in the list.
  mlir::LogicalResult verify(Type ty);

  void print(mlir::DialectAsmPrinter &printer);
};

/// Match all the TypeConstrants in the list.
class AllOfType : public SpecType<AllOfType, SpecTypes::AllOf,
                                  detail::TypeListStorage> {
public:
  using Base::Base;

  static AllOfType get(llvm::ArrayRef<Type> tys);
  static AllOfType getChecked(mlir::Location loc, llvm::ArrayRef<Type> tys);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, llvm::ArrayRef<Type> tys);
  mlir::LogicalResult verify(Type ty);

  void print(mlir::DialectAsmPrinter &printer);
};

/// Match any IntegerType.
class AnyIntegerType
    : public SimpleType<AnyIntegerType, SpecTypes::AnyInteger> {
public:
  using Base::Base;
  inline mlir::LogicalResult verify(Type ty) {
    return mlir::success(ty.isa<mlir::IntegerType>());
  }
};

/// Match any IntegerType of specified width.
class AnyIType : public SpecType<AnyIType, SpecTypes::AnyI,
                                 detail::WidthStorage> {
public:
  using Base::Base;

  static AnyIType get(unsigned width, mlir::MLIRContext *ctx);
  static AnyIType getChecked(mlir::Location loc, unsigned width);

  /// Width must be one of [1, 8, 16, 32, 64].
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, unsigned width);
  /// Check Type is an integer of specified width.
  mlir::LogicalResult verify(Type ty);
};

/// Match any IntegerType of the specified widths.
class AnyIntOfWidthsType
    : public SpecType<AnyIntOfWidthsType, SpecTypes::AnyIntOfWidths,
                      detail::WidthListStorage> {
public:
  using Base::Base;

  static AnyIntOfWidthsType get(
      llvm::ArrayRef<unsigned> widths, mlir::MLIRContext *ctx);
  static AnyIntOfWidthsType getChecked(mlir::Location loc,
                                       llvm::ArrayRef<unsigned> widths);

  /// Each width must be one of [1, 8, 16, 32, 64];
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, llvm::ArrayRef<unsigned> widths);
  /// Check Type is an integer of one of the specified widths.
  mlir::LogicalResult verify(Type ty);
};

/// Match any signless integer.
class AnySignlessIntegerType : public SimpleType<
    AnySignlessIntegerType, SpecTypes::AnySignlessInteger> {
public:
  using Base::Base;
  inline mlir::LogicalResult verify(Type ty) {
    return mlir::success(ty.isSignlessInteger());
  }
};

/// Match a signless integer of a specified width.
class IType : public SpecType<IType, SpecTypes::I, detail::WidthStorage> {
public:
  using Base::Base;

  static IType get(unsigned width, mlir::MLIRContext *ctx);
  static IType getChecked(mlir::Location loc, unsigned width);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, unsigned width);
  mlir::LogicalResult verify(Type ty);
};

/// Match a signless integer of one of the specified widths.
class SignlessIntOfWidthsType
    : public SpecType<SignlessIntOfWidthsType, SpecTypes::SignlessIntOfWidths,
                      detail::WidthListStorage> {
public:
  using Base::Base;

  static SignlessIntOfWidthsType get(
      llvm::ArrayRef<unsigned> widths, mlir::MLIRContext *ctx);
  static SignlessIntOfWidthsType getChecked(
      mlir::Location loc, llvm::ArrayRef<unsigned> widths);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, llvm::ArrayRef<unsigned> widths);
  mlir::LogicalResult verify(Type ty);
};

/// Match any signed integer.
class AnySignedIntegerType
    : public SimpleType<AnySignedIntegerType,
                        SpecTypes::AnySignedInteger> {
public:
  using Base::Base;
  inline mlir::LogicalResult verify(Type ty) {
    return mlir::success(ty.isSignedInteger());
  }
};

/// Match any signed integer of the specified width;
class SIType : public SpecType<SIType, SpecTypes::SI, detail::WidthStorage> {
public:
  using Base::Base;

  static SIType get(unsigned width, mlir::MLIRContext *ctx);
  static SIType getChecked(mlir::Location loc, unsigned width);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, unsigned width);
  mlir::LogicalResult verify(Type ty);
};

/// Match any signed integer of the specified widths.
class SignedIntOfWidthsType
    : public SpecType<SignedIntOfWidthsType, SpecTypes::SignedIntOfWidths,
                      detail::WidthListStorage> {
public:
  using Base::Base;

  static SignedIntOfWidthsType get(
      llvm::ArrayRef<unsigned> widths, mlir::MLIRContext *ctx);
  static SignedIntOfWidthsType getChecked(
      mlir::Location, llvm::ArrayRef<unsigned> widths);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location, llvm::ArrayRef<unsigned> widths);
  mlir::LogicalResult verify(Type ty);
};

/// Match any unsigned integer.
class AnyUnsignedIntegerType
    : public SimpleType<AnyUnsignedIntegerType,
                        SpecTypes::AnyUnsignedInteger> {
public:
  using Base::Base;
  inline mlir::LogicalResult verify(Type ty) {
    return mlir::success(ty.isUnsignedInteger());
  }
};

/// Match any unsigned integer of the specified width.
class UIType : public SpecType<UIType, SpecTypes::UI, detail::WidthStorage> {
public:
  using Base::Base;

  static UIType get(unsigned width, mlir::MLIRContext *ctx);
  static UIType getChecked(mlir::Location loc, unsigned width);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, unsigned width);
  mlir::LogicalResult verify(Type ty);
};

/// Match any unsigned integer of the specified widths.
class UnsignedIntOfWidthsType
    : public SpecType<UnsignedIntOfWidthsType, SpecTypes::UnsignedIntOfWidths,
                      detail::WidthListStorage> {
public:
  using Base::Base;

  static UnsignedIntOfWidthsType get(
      llvm::ArrayRef<unsigned> widths, mlir::MLIRContext *ctx);
  static UnsignedIntOfWidthsType getChecked(
      mlir::Location loc, llvm::ArrayRef<unsigned> widths);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, llvm::ArrayRef<unsigned> widths);
  mlir::LogicalResult verify(Type ty);
};

/// Match an index type.
class IndexType : public SimpleType<IndexType, SpecTypes::Index> {
public:
  using Base::Base;
  mlir::LogicalResult verify(Type ty) {
    return mlir::success(ty.isa<mlir::IndexType>());
  }
};

/// Match any floating point type.
class AnyFloatType : public SimpleType<AnyFloatType,
                                       SpecTypes::AnyFloat> {
public:
  using Base::Base;
  mlir::LogicalResult verify(Type ty) {
    return mlir::success(ty.isa<mlir::FloatType>());
  }
};

/// Match a float of the specified width.
class FType : public SpecType<FType, SpecTypes::F, detail::WidthStorage> {
public:
  using Base::Base;

  static FType get(unsigned width, mlir::MLIRContext *ctx);
  static FType getChecked(mlir::Location loc, unsigned width);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, unsigned width);
  mlir::LogicalResult verify(Type ty);
};

/// Match a float of the specified widths.
class FloatOfWidthsType
    : public SpecType<FloatOfWidthsType, SpecTypes::FloatOfWidths,
                      detail::WidthListStorage> {
public:
  using Base::Base;

  static FloatOfWidthsType get(
      llvm::ArrayRef<unsigned> widths, mlir::MLIRContext *ctx);
  static FloatOfWidthsType getChecked(
      mlir::Location loc, llvm::ArrayRef<unsigned> widths);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, llvm::ArrayRef<unsigned> widths);
  mlir::LogicalResult verify(Type ty);
};

/// Match BF16.
class BF16Type : public SimpleType<BF16Type, SpecTypes::BF16> {
public:
  using Base::Base;
  mlir::LogicalResult verify(Type ty) {
    return mlir::success(ty.isBF16());
  }
};

/// Match any ComplexType.
class AnyComplexType : public SimpleType<AnyComplexType,
                                         SpecTypes::AnyComplex> {
public:
  using Base::Base;
  mlir::LogicalResult verify(Type ty) {
    return mlir::success(ty.isa<mlir::ComplexType>());
  }
};

/// Match a ComplexType of an element type.
class ComplexType : public SpecType<ComplexType, SpecTypes::Complex,
                                    detail::OneTypeStorage> {
public:
  using Base::Base;

  static ComplexType get(Type elTy);
  static ComplexType getChecked(mlir::Location loc, Type elTy);
  mlir::LogicalResult verify(Type ty);

  static Type parse(mlir::DialectAsmParser &parser);
  void print(mlir::DialectAsmPrinter &printer);
};

/// Match an opaque type based on name.
class OpaqueType : public SpecType<OpaqueType, SpecTypes::Opaque,
                                   detail::OpaqueTypeStorage> {
public:
  using Base::Base;

  static OpaqueType get(llvm::StringRef dialectName, llvm::StringRef typeName,
                        mlir::MLIRContext *ctx);
  static OpaqueType getChecked(mlir::Location loc, llvm::StringRef dialectName,
                               llvm::StringRef typeName);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc,
      llvm::StringRef dialectName, llvm::StringRef typeName);
  mlir::LogicalResult verify(Type ty);

  static Type parse(mlir::DialectAsmParser &parser);
  void print(mlir::DialectAsmPrinter &printer);
};

/// Match a FunctionType.
class FunctionType : public SimpleType<FunctionType, SpecTypes::Function> {
public:
  using Base::Base;
  inline mlir::LogicalResult verify(Type ty) {
    return mlir::success(ty.isa<mlir::FunctionType>());
  }
};

/// Variadic values.
class VariadicType : public SpecType<VariadicType, SpecTypes::Variadic,
                                     detail::OneTypeStorage> {
public:
  using Base::Base;

  static VariadicType get(mlir::Type ty);
  static VariadicType getChecked(mlir::Location loc, mlir::Type ty);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, mlir::Type ty);
  mlir::LogicalResult verify(mlir::Type ty);

  static Type parse(mlir::DialectAsmParser &parser);
  void print(mlir::DialectAsmPrinter &printer);
};

/// Optional values.

/// TODO Container types (vectors, tensors, etc.), memref types, tuples.
/// TODO variadic and optional types.

} // end namespace dmc
