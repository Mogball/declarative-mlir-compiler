#pragma once

#include "SpecTypeImplementation.h"
#include "SpecTypeDetail.h"

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
struct IsaTypeStorage;
} // end namespace detail

/// Match any type.
class AnyType : public SimpleType<AnyType, SpecTypes::Any> {
public:
  using Base::Base;
  static llvm::StringLiteral getTypeName() { return "Any"; }
  inline mlir::LogicalResult verify(Type) { return mlir::success(); }
};

/// Match NoneType.
class NoneType : public SimpleType<NoneType, SpecTypes::None> {
public:
  using Base::Base;
  static llvm::StringLiteral getTypeName() { return "None"; }
  inline mlir::LogicalResult verify(Type ty) {
    return mlir::success(ty.isa<mlir::NoneType>());
  }
};

/// Match any Type in a list. The type list cannot be empty.
class AnyOfType : public SpecType<AnyOfType, SpecTypes::AnyOf,
                                  detail::TypeListStorage> {
public:
  using Base::Base;
  static llvm::StringLiteral getTypeName() { return "AnyOf"; }

  static AnyOfType get(llvm::ArrayRef<Type> tys);
  static AnyOfType getChecked(mlir::Location loc, llvm::ArrayRef<Type> tys);

  /// Type list cannot be empty.
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, llvm::ArrayRef<Type> tys);
  /// Check Type is in the list.
  mlir::LogicalResult verify(Type ty);

  static Type parse(mlir::DialectAsmParser &parser);
  void print(mlir::DialectAsmPrinter &printer);
};

/// Match all the TypeConstrants in the list.
class AllOfType : public SpecType<AllOfType, SpecTypes::AllOf,
                                  detail::TypeListStorage> {
public:
  using Base::Base;
  static llvm::StringLiteral getTypeName() { return "AllOf"; }

  static AllOfType get(llvm::ArrayRef<Type> tys);
  static AllOfType getChecked(mlir::Location loc, llvm::ArrayRef<Type> tys);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, llvm::ArrayRef<Type> tys);
  mlir::LogicalResult verify(Type ty);

  static Type parse(mlir::DialectAsmParser &parser);
  void print(mlir::DialectAsmPrinter &printer);
};

/// Match any IntegerType.
class AnyIntegerType
    : public SimpleType<AnyIntegerType, SpecTypes::AnyInteger> {
public:
  using Base::Base;
  static llvm::StringLiteral getTypeName() { return "AnyInteger"; }
  inline mlir::LogicalResult verify(Type ty) {
    return mlir::success(ty.isa<mlir::IntegerType>());
  }
};

/// Match any IntegerType of specified width.
class AnyIType : public IntegerWidthType<AnyIType, SpecTypes::AnyI> {
public:
  using Base::Base;
  static llvm::StringLiteral getTypeName() { return "AnyI"; }
  /// Check Type is an integer of specified width.
  mlir::LogicalResult verify(Type ty);
};

/// Match any IntegerType of the specified widths.
class AnyIntOfWidthsType : public IntegerTypeOfWidths<
    AnyIntOfWidthsType, SpecTypes::AnyIntOfWidths> {
public:
  using Base::Base;
  static llvm::StringLiteral getTypeName() { return "AnyIntOfWidths"; }
  /// Check Type is an integer of one of the specified widths.
  mlir::LogicalResult verify(Type ty);
};

/// Match any signless integer.
class AnySignlessIntegerType : public SimpleType<
    AnySignlessIntegerType, SpecTypes::AnySignlessInteger> {
public:
  using Base::Base;
  static llvm::StringLiteral getTypeName() { return "AnySignlessInteger"; }
  inline mlir::LogicalResult verify(Type ty) {
    return mlir::success(ty.isSignlessInteger());
  }
};

/// Match a signless integer of a specified width.
class IType : public IntegerWidthType<IType, SpecTypes::I> {
public:
  using Base::Base;
  static llvm::StringLiteral getTypeName() { return "I"; }
  mlir::LogicalResult verify(Type ty);
};

/// Match a signless integer of one of the specified widths.
class SignlessIntOfWidthsType : public IntegerTypeOfWidths<
    SignlessIntOfWidthsType, SpecTypes::SignlessIntOfWidths> {
public:
  using Base::Base;
  static llvm::StringLiteral getTypeName() { return "SignlessIntOfWidths"; }
  mlir::LogicalResult verify(Type ty);
};

/// Match any signed integer.
class AnySignedIntegerType
    : public SimpleType<AnySignedIntegerType,
                        SpecTypes::AnySignedInteger> {
public:
  using Base::Base;
  static llvm::StringLiteral getTypeName() { return "AnySignedInteger"; }
  inline mlir::LogicalResult verify(Type ty) {
    return mlir::success(ty.isSignedInteger());
  }
};

/// Match any signed integer of the specified width;
class SIType : public IntegerWidthType<SIType, SpecTypes::SI> {
public:
  using Base::Base;
  static llvm::StringLiteral getTypeName() { return "SI"; }
  mlir::LogicalResult verify(Type ty);
};

/// Match any signed integer of the specified widths.
class SignedIntOfWidthsType : public IntegerTypeOfWidths<
    SignedIntOfWidthsType, SpecTypes::SignedIntOfWidths> {
public:
  using Base::Base;
  static llvm::StringLiteral getTypeName() { return "SignedIntOfWidths"; }
  mlir::LogicalResult verify(Type ty);
};

/// Match any unsigned integer.
class AnyUnsignedIntegerType
    : public SimpleType<AnyUnsignedIntegerType,
                        SpecTypes::AnyUnsignedInteger> {
public:
  using Base::Base;
  static llvm::StringLiteral getTypeName() { return "AnyUnsignedInteger"; }
  inline mlir::LogicalResult verify(Type ty) {
    return mlir::success(ty.isUnsignedInteger());
  }
};

/// Match any unsigned integer of the specified width.
class UIType : public IntegerWidthType<UIType, SpecTypes::UI> {
public:
  using Base::Base;
  static llvm::StringLiteral getTypeName() { return "UI"; }
  mlir::LogicalResult verify(Type ty);
};

/// Match any unsigned integer of the specified widths.
class UnsignedIntOfWidthsType : public IntegerTypeOfWidths<
    UnsignedIntOfWidthsType, SpecTypes::UnsignedIntOfWidths> {
public:
  using Base::Base;
  static llvm::StringLiteral getTypeName() { return "UnsignedIntOfWidths"; }
  mlir::LogicalResult verify(Type ty);
};

/// Match an index type.
class IndexType : public SimpleType<IndexType, SpecTypes::Index> {
public:
  using Base::Base;
  static llvm::StringLiteral getTypeName() { return "Index"; }
  mlir::LogicalResult verify(Type ty) {
    return mlir::success(ty.isa<mlir::IndexType>());
  }
};

/// Match any floating point type.
class AnyFloatType : public SimpleType<AnyFloatType,
                                       SpecTypes::AnyFloat> {
public:
  using Base::Base;
  static llvm::StringLiteral getTypeName() { return "AnyFloat"; }
  mlir::LogicalResult verify(Type ty) {
    return mlir::success(ty.isa<mlir::FloatType>());
  }
};

/// Match a float of the specified width.
class FType : public NumericWidthType<FType, SpecTypes::F> {
public:
  using Base::Base;
  static llvm::StringLiteral getTypeName() { return "F"; }
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, unsigned width);
  mlir::LogicalResult verify(Type ty);
};

/// Match a float of the specified widths.
class FloatOfWidthsType : public NumericTypeOfWidths<
    FloatOfWidthsType, SpecTypes::FloatOfWidths> {
public:
  using Base::Base;
  static llvm::StringLiteral getTypeName() { return "FloatOfWidths"; }
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, llvm::ArrayRef<unsigned> widths);
  mlir::LogicalResult verify(Type ty);
};

/// Match BF16.
class BF16Type : public SimpleType<BF16Type, SpecTypes::BF16> {
public:
  using Base::Base;
  static llvm::StringLiteral getTypeName() { return "BF16"; }
  mlir::LogicalResult verify(Type ty) {
    return mlir::success(ty.isBF16());
  }
};

/// Match any ComplexType.
class AnyComplexType : public SimpleType<AnyComplexType,
                                         SpecTypes::AnyComplex> {
public:
  using Base::Base;
  static llvm::StringLiteral getTypeName() { return "AnyComplex"; }
  mlir::LogicalResult verify(Type ty) {
    return mlir::success(ty.isa<mlir::ComplexType>());
  }
};

/// Match a ComplexType of an element type.
class ComplexType : public SpecType<ComplexType, SpecTypes::Complex,
                                    detail::OneTypeStorage> {
public:
  using Base::Base;
  static llvm::StringLiteral getTypeName() { return "Complex"; }

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
  static llvm::StringLiteral getTypeName() { return "Opaque"; }

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
  static llvm::StringLiteral getTypeName() { return "Function"; }
  inline mlir::LogicalResult verify(Type ty) {
    return mlir::success(ty.isa<mlir::FunctionType>());
  }
};

/// Variadic values.
class VariadicType : public SpecType<VariadicType, SpecTypes::Variadic,
                                     detail::OneTypeStorage> {
public:
  using Base::Base;
  static llvm::StringLiteral getTypeName() { return "Variadic"; }

  static VariadicType get(mlir::Type ty);
  static VariadicType getChecked(mlir::Location loc, mlir::Type ty);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, mlir::Type ty);
  mlir::LogicalResult verify(mlir::Type ty);

  static Type parse(mlir::DialectAsmParser &parser);
  void print(mlir::DialectAsmPrinter &printer);
};

/// Check if a type range contains a variadic type.
template <typename TypeRange>
inline static bool hasVariadicValues(const TypeRange &tys) {
  return llvm::count_if(tys,
      [](mlir::Type ty) { return ty.isa<VariadicType>(); });
}

/// Match a dynamic type based on kind. For example, to match a type
///
/// dmc.Dialect @MyDialect {
///   dmc.Type @MyType<i32>
/// }
///
/// With any parameter values, use !dmc.Isa<@MyDialect::@MyType>.
class IsaType : public SpecType<IsaType, SpecTypes::Isa,
                                detail::IsaTypeStorage> {
public:
  using Base::Base;
  static llvm::StringLiteral getTypeName() { return "Isa"; }

  static IsaType get(mlir::SymbolRefAttr typeRef);
  static IsaType getChecked(mlir::Location loc, mlir::SymbolRefAttr typeRef);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, mlir::SymbolRefAttr typeRef);
  mlir::LogicalResult verify(Type ty);

  static Type parse(mlir::DialectAsmParser &parser);
  void print(mlir::DialectAsmPrinter &printer);
};

/// TODO Container types (vectors, tensors, etc.), memref types, tuples.

} // end namespace dmc
