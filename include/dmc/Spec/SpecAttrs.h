#pragma once

#include "SpecAttrDetail.h"
#include "SpecAttrBase.h"
#include "SpecTypes.h"

#include <mlir/IR/StandardTypes.h>

namespace dmc {

namespace detail {
struct AttrListStorage;
struct OneTypeAttrStorage;
struct DimensionAttrStorage;
struct DefaultAttrStorage;
struct IsaAttrStorage;
struct PyAttrStorage;
} // end namespace detail

class AnyAttr : public SimpleAttr<AnyAttr, SpecAttrs::Any> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "Any"; }
  inline mlir::LogicalResult verify(Attribute) { return mlir::success(); }
};

class BoolAttr : public SimpleAttr<BoolAttr, SpecAttrs::Bool> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "Bool"; }
  inline mlir::LogicalResult verify(Attribute attr) {
    return mlir::success(attr.isa<mlir::BoolAttr>());
  }
};

class IndexAttr : public SimpleAttr<IndexAttr, SpecAttrs::Index> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "Index"; }
  inline mlir::LogicalResult verify(Attribute attr) {
    return mlir::success(attr.isa<mlir::IntegerAttr>() &&
        attr.cast<mlir::IntegerAttr>().getType().isa<mlir::IndexType>());
  }
};

class APIntAttr : public SimpleAttr<APIntAttr, SpecAttrs::APInt> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "APInt"; }
  inline mlir::LogicalResult verify(Attribute attr) {
    return mlir::success(attr.isa<mlir::IntegerAttr>());
  }
};

class AnyIAttr : public TypedAttrBase<AnyIAttr,
                 SpecAttrs::AnyI, mlir::IntegerAttr, AnyIType> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "AnyI"; }
};

class IAttr : public TypedAttrBase<IAttr,
              SpecAttrs::I, mlir::IntegerAttr, IType> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "I"; }
};

class SIAttr : public TypedAttrBase<SIAttr,
               SpecAttrs::SI, mlir::IntegerAttr, SIType> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "SI"; }
};

class UIAttr : public TypedAttrBase<UIAttr,
               SpecAttrs::UI, mlir::IntegerAttr, UIType> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "UI"; }
};

class FAttr : public TypedAttrBase<FAttr,
              SpecAttrs::F, mlir::FloatAttr, FType> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "F"; }
};

class StringAttr : public SimpleAttr<StringAttr, SpecAttrs::String> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "String"; }
  inline mlir::LogicalResult verify(Attribute attr) {
    return mlir::success(attr.isa<mlir::StringAttr>());
  }
};

class TypeAttr : public SimpleAttr<TypeAttr, SpecAttrs::Type> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "Type"; }
  inline mlir::LogicalResult verify(Attribute attr) {
    return mlir::success(attr.isa<mlir::TypeAttr>());
  }
};

class UnitAttr : public SimpleAttr<UnitAttr, SpecAttrs::Unit> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "Unit"; }
  inline mlir::LogicalResult verify(Attribute attr) {
    return mlir::success(attr.isa<mlir::UnitAttr>());
  }
};

class DictionaryAttr : public SimpleAttr<DictionaryAttr,
                                         SpecAttrs::Dictionary> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "Dictionary"; }
  inline mlir::LogicalResult verify(Attribute attr) {
    return mlir::success(attr.isa<mlir::DictionaryAttr>());
  }
};

class ElementsAttr : public SimpleAttr<ElementsAttr,
                                       SpecAttrs::Elements> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "Elements"; }
  inline mlir::LogicalResult verify(Attribute attr) {
    return mlir::success(attr.isa<mlir::ElementsAttr>());
  }
};

class DenseElementsAttr : public SimpleAttr<DenseElementsAttr,
                                            SpecAttrs::DenseElements> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "DenseElements"; }
  inline mlir::LogicalResult verify(Attribute attr) {
    return mlir::success(attr.isa<mlir::DenseElementsAttr>());
  }
};

/// Compose this attribute constraint with type constraints to achieve typed
/// elements. For example,
///
/// Alias @F64ElementsAttr -> #And<#ElementsOf<f64>, #DenseElements>
/// Alias @RankedI32ElementsAttr -> #And<#ElementsOf<!I<32>>, #DenseElements,
///                                      #RankedElements<[2, 2]>>
class ElementsOfAttr : public SpecAttr<ElementsOfAttr, SpecAttrs::ElementsOf,
                                       detail::OneTypeAttrStorage> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "ElementsOf"; }

  static ElementsOfAttr get(mlir::Type elTy);
  mlir::LogicalResult verify(Attribute attr);

  static Attribute parse(mlir::DialectAsmParser &parser);
  void print(mlir::DialectAsmPrinter &printer);
};

class RankedElementsAttr : public SpecAttr<
    RankedElementsAttr, SpecAttrs::RankedElements,
    detail::DimensionAttrStorage> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "Ranked"; }

  static RankedElementsAttr getChecked(mlir::Location loc,
                                       llvm::ArrayRef<int64_t> dims);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, llvm::ArrayRef<int64_t> dims);
  mlir::LogicalResult verify(Attribute attr);

  static Attribute parse(mlir::DialectAsmParser &parser);
  void print(mlir::DialectAsmPrinter &printer);
};

class StringElementsAttr : public SimpleAttr<StringElementsAttr,
                                             SpecAttrs::StringElements> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "StringElements"; }
  inline mlir::LogicalResult verify(Attribute attr) {
    return mlir::success(attr.isa<mlir::DenseStringElementsAttr>());
  }
};

class ArrayAttr : public SimpleAttr<ArrayAttr, SpecAttrs::Array> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "Array"; }
  inline mlir::LogicalResult verify(Attribute attr) {
    return mlir::success(attr.isa<mlir::ArrayAttr>());
  }
};

/// An array attribute with an attribute constraint applied to each value.
class ArrayOfAttr : public SpecAttr<ArrayOfAttr, SpecAttrs::ArrayOf,
                                    detail::OneAttrStorage> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "ArrayOf"; }

  /// An array of a constant attribute makes no sense, so assert that the
  /// attribute is a SpecAttr constraint.
  static ArrayOfAttr getChecked(mlir::Location loc, Attribute constraint);
  static mlir::LogicalResult verifyConstructionInvariants(mlir::Location,
                                                          Attribute constraint);
  mlir::LogicalResult verify(Attribute attr);

  static Attribute parse(mlir::DialectAsmParser &parser);
  void print(mlir::DialectAsmPrinter &printer);
};

class SymbolRefAttr : public SimpleAttr<SymbolRefAttr, SpecAttrs::SymbolRef> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "SymbolRef"; }
  inline mlir::LogicalResult verify(Attribute attr) {
    return mlir::success(attr.isa<mlir::SymbolRefAttr>());
  }
};

class FlatSymbolRefAttr : public SimpleAttr<FlatSymbolRefAttr,
                                            SpecAttrs::FlatSymbolRef> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "FlatSymbolRef"; }
  inline mlir::LogicalResult verify(Attribute attr) {
    return mlir::success(attr.isa<mlir::FlatSymbolRefAttr>());
  }
};

class ConstantAttr : public SpecAttr<ConstantAttr, SpecAttrs::Constant,
                                     detail::OneAttrStorage> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "Constant"; }

  static ConstantAttr get(Attribute attr);
  mlir::LogicalResult verify(Attribute attr);

  static Attribute parse(mlir::DialectAsmParser &parser);
  void print(mlir::DialectAsmPrinter &printer);
};

class AnyOfAttr : public SpecAttr<AnyOfAttr, SpecAttrs::AnyOf,
                                  detail::AttrListStorage> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "AnyOf"; }

  static AnyOfAttr getChecked(mlir::Location loc,
                                 llvm::ArrayRef<Attribute> attrs);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, llvm::ArrayRef<Attribute> attrs);
  mlir::LogicalResult verify(Attribute attr);

  static Attribute parse(mlir::DialectAsmParser &parser);
  void print(mlir::DialectAsmPrinter &printer);
};

class AllOfAttr : public SpecAttr<AllOfAttr, SpecAttrs::AllOf,
                                  detail::AttrListStorage> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "AllOf"; }

  static AllOfAttr getChecked(mlir::Location loc,
                                 llvm::ArrayRef<Attribute> attrs);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, llvm::ArrayRef<Attribute> attrs);
  mlir::LogicalResult verify(Attribute attr);

  static Attribute parse(mlir::DialectAsmParser &parser);
  void print(mlir::DialectAsmPrinter &printer);
};

class OfTypeAttr : public SpecAttr<OfTypeAttr, SpecAttrs::OfType,
                                   detail::OneTypeAttrStorage> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "OfType"; }

  static OfTypeAttr get(mlir::Type ty);
  mlir::LogicalResult verify(Attribute attr);

  static Attribute parse(mlir::DialectAsmParser &parser);
  void print(mlir::DialectAsmPrinter &printer);
};

/// An optional attribute. The generated Attribute constraint will not check
/// for the Attribute's presence, but will apply the constraint if present.
class OptionalAttr : public SpecAttr<OptionalAttr, SpecAttrs::Optional,
                                     detail::OneAttrStorage> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "Optional"; }

  static OptionalAttr get(Attribute baseAttr);
  mlir::LogicalResult verify(Attribute attr);

  static Attribute parse(mlir::DialectAsmParser &parser);
  void print(mlir::DialectAsmPrinter &printer);
};

/// A default-valued attribute. During Op post-construction, if this attribute
/// was not provided, the default value will be used.
/// TODO use default attributes in op builders.
class DefaultAttr : public SpecAttr<DefaultAttr, SpecAttrs::Default,
                                    detail::DefaultAttrStorage> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "Default"; }

  static DefaultAttr get(Attribute baseAttr, Attribute defaultAttr);
  mlir::LogicalResult verify(Attribute attr);

  static Attribute parse(mlir::DialectAsmParser &parser);
  void print(mlir::DialectAsmPrinter &printer);

  /// Get the default value.
  Attribute getDefaultValue();
};

class IsaAttr : public SpecAttr<IsaAttr, SpecAttrs::Isa,
                                detail::IsaAttrStorage> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "Isa"; }

  static IsaAttr getChecked(mlir::Location loc, mlir::SymbolRefAttr attrRef);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, mlir::SymbolRefAttr attrRef);
  mlir::LogicalResult verify(Attribute attr);

  static Attribute parse(mlir::DialectAsmParser &parser);
  void print(mlir::DialectAsmPrinter &printer);
};

/// A generic Python attribute constraint. See `PyType`.
class PyAttr : public SpecAttr<PyAttr, SpecAttrs::Py, detail::PyAttrStorage> {
public:
  using Base::Base;
  static llvm::StringLiteral getAttrName() { return "Py"; }

  static PyAttr getChecked(mlir::Location loc, llvm::StringRef expr);
  mlir::LogicalResult verify(Attribute attr);

  static Attribute parse(mlir::DialectAsmParser &parser);
  void print(mlir::DialectAsmPrinter &printer);
};

/// TODO struct attributes, enum attributes?

} // end namespace dmc
