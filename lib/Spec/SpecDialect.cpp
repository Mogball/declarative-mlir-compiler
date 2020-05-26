#include "dmc/Spec/SpecDialect.h"
#include "dmc/Spec/SpecOps.h"
#include "dmc/Spec/SpecRegion.h"
#include "dmc/Spec/SpecTypeSwitch.h"
#include "dmc/Spec/SpecAttrSwitch.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <llvm/ADT/StringSwitch.h>

using namespace mlir;

namespace dmc {

/// SpecDialect.
SpecDialect::SpecDialect(MLIRContext *ctx)
    : Dialect{getDialectNamespace(), ctx} {
  addOperations<
      DialectOp, DialectTerminatorOp, OperationOp, TypeOp, AttributeOp,
      AliasOp
  >();
  addTypes<
      AnyType, NoneType, AnyOfType, AllOfType,
      AnyIntegerType, AnyIType, AnyIntOfWidthsType,
      AnySignlessIntegerType, IType, SignlessIntOfWidthsType,
      AnySignedIntegerType, SIType, SignedIntOfWidthsType,
      AnyUnsignedIntegerType, UIType, UnsignedIntOfWidthsType,
      IndexType, AnyFloatType, FType, FloatOfWidthsType, BF16Type,
      AnyComplexType, ComplexType, OpaqueType,
      FunctionType, VariadicType, IsaType
  >();
  addAttributes<
      AnyAttr, BoolAttr, IndexAttr, APIntAttr,
      AnyIAttr, IAttr, SIAttr, UIAttr, FAttr,
      StringAttr, TypeAttr, UnitAttr,
      DictionaryAttr, ElementsAttr, ArrayAttr,
      SymbolRefAttr, FlatSymbolRefAttr,
      ConstantAttr, AnyOfAttr, AllOfAttr, OfTypeAttr,
      OptionalAttr, DefaultAttr, IsaAttr,

      AnyRegion, SizedRegion, IsolatedFromAboveRegion, VariadicRegion
  >();
}

namespace {

template <typename BaseT>
Type parseTypeList(DialectAsmParser &parser) {
  // *-of-type ::= `AnyOf` `<` type(`,` type)* `>`
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  if (parser.parseLess())
    return Type{};
  SmallVector<Type, 4> baseTypes;
  do {
    Type baseType;
    if (parser.parseType(baseType))
      return Type{};
    baseTypes.push_back(baseType);
  } while (!parser.parseOptionalComma());
  if (parser.parseGreater())
    return Type{};
  return BaseT::getChecked(loc, baseTypes);
}

} // end anonymous namespace

/// Type parsing.
Type SpecDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeName;
  if (parser.parseKeyword(&typeName))
    return Type{};
  auto kind = llvm::StringSwitch<unsigned>(typeName)
    .Case(AnyType::getTypeName(), SpecTypes::Any)
    .Case(NoneType::getTypeName(), SpecTypes::None)
    .Case(AnyOfType::getTypeName(), SpecTypes::AnyOf)
    .Case(AllOfType::getTypeName(), SpecTypes::AllOf)
    .Case(AnyIntegerType::getTypeName(), SpecTypes::AnyInteger)
    .Case(AnyIType::getTypeName(), SpecTypes::AnyI)
    .Case(AnyIntOfWidthsType::getTypeName(), SpecTypes::AnyIntOfWidths)
    .Case(AnySignlessIntegerType::getTypeName(), SpecTypes::AnySignlessInteger)
    .Case(IType::getTypeName(), SpecTypes::I)
    .Case(SignlessIntOfWidthsType::getTypeName(), SpecTypes::SignlessIntOfWidths)
    .Case(AnySignedIntegerType::getTypeName(), SpecTypes::AnySignedInteger)
    .Case(SIType::getTypeName(), SpecTypes::SI)
    .Case(SignedIntOfWidthsType::getTypeName(), SpecTypes::SignedIntOfWidths)
    .Case(AnyUnsignedIntegerType::getTypeName(), SpecTypes::AnyUnsignedInteger)
    .Case(UIType::getTypeName(), SpecTypes::UI)
    .Case(UnsignedIntOfWidthsType::getTypeName(), SpecTypes::UnsignedIntOfWidths)
    .Case(IndexType::getTypeName(), SpecTypes::Index)
    .Case(AnyFloatType::getTypeName(), SpecTypes::AnyFloat)
    .Case(FType::getTypeName(), SpecTypes::F)
    .Case(FloatOfWidthsType::getTypeName(), SpecTypes::FloatOfWidths)
    .Case(BF16Type::getTypeName(), SpecTypes::BF16)
    .Case(AnyComplexType::getTypeName(), SpecTypes::AnyComplex)
    .Case(ComplexType::getTypeName(), SpecTypes::Complex)
    .Case(OpaqueType::getTypeName(), SpecTypes::Opaque)
    .Case(FunctionType::getTypeName(), SpecTypes::Function)
    .Case(VariadicType::getTypeName(), SpecTypes::Variadic)
    .Case(IsaType::getTypeName(), SpecTypes::Isa)
    .Default(SpecTypes::NUM_TYPES);

  if (kind == SpecTypes::NUM_TYPES) {
    parser.emitError(parser.getCurrentLocation(), "unknown type constraint");
    return Type{};
  }
  ParseAction<Type, DialectAsmParser> action{parser};
  return SpecTypes::kindSwitch(action, kind);
}

Type AnyOfType::parse(DialectAsmParser &parser) {
  return parseTypeList<AnyOfType>(parser);
}

Type AllOfType::parse(DialectAsmParser &parser) {
  return parseTypeList<AllOfType>(parser);
}

Type ComplexType::parse(DialectAsmParser &parser) {
  // `Complex` `<` type `>`
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  Type elBaseType;
  if (parser.parseLess() || parser.parseType(elBaseType) ||
      parser.parseGreater())
    return Type{};
  return ComplexType::getChecked(loc, elBaseType);
}

Type OpaqueType::parse(DialectAsmParser &parser) {
  // `Opaque` `<` string `,` string `>`
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  mlir::StringAttr dialectNameAttr, typeNameAttr;
  if (parser.parseLess() || parser.parseAttribute(dialectNameAttr) ||
      parser.parseComma() || parser.parseAttribute(typeNameAttr) ||
      parser.parseGreater())
    return Type{};
  return OpaqueType::getChecked(loc, dialectNameAttr.getValue(),
                                typeNameAttr.getValue());
}

Type VariadicType::parse(DialectAsmParser &parser) {
  // `Variadic` `<`type `>`
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  Type baseTy;
  if (parser.parseLess() || parser.parseType(baseTy) ||
      parser.parseGreater())
    return Type{};
  return VariadicType::getChecked(loc, baseTy);
}

Type IsaType::parse(DialectAsmParser &parser) {
  // `Isa` `<` sym-ref `>`
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  mlir::SymbolRefAttr typeRef;
  if (parser.parseLess() || parser.parseAttribute(typeRef) ||
      parser.parseGreater())
    return Type{};
  return IsaType::getChecked(loc, typeRef);
}

/// Attribute parsing.
namespace {

template <typename BaseT>
Attribute parseAttrList(DialectAsmParser &parser) {
  // *of-attr ::= `*Of` `<` attr(`,` attr)* `>`
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  if (parser.parseLess())
    return Attribute{};
  SmallVector<Attribute, 4> baseAttrs;
  do {
    Attribute baseAttr;
    if (parser.parseAttribute(baseAttr))
      return Attribute{};
    baseAttrs.push_back(baseAttr);
  } while (!parser.parseOptionalComma());
  if (parser.parseGreater())
    return Attribute{};
  return BaseT::getChecked(loc, baseAttrs);
}

} // end anonymous namespace

Attribute SpecDialect::parseAttribute(DialectAsmParser &parser,
                                      Type type) const {
  /// TODO supported for typed Spec attributes. Currently, either a null type
  /// or NoneType is passed during parsing.
  if (type && !type.isa<mlir::NoneType>()) {
    parser.emitError(parser.getCurrentLocation(),
                     "spec attribute cannot have a type: ") << type;
    return Attribute{};
  }
  StringRef attrName;
  if (parser.parseKeyword(&attrName))
    return Attribute{};
  auto kind = llvm::StringSwitch<unsigned>(attrName)
    .Case(AnyAttr::getAttrName(), SpecAttrs::Any)
    .Case(BoolAttr::getAttrName(), SpecAttrs::Bool)
    .Case(IndexAttr::getAttrName(), SpecAttrs::Index)
    .Case(APIntAttr::getAttrName(), SpecAttrs::APInt)
    .Case(AnyIAttr::getAttrName(), SpecAttrs::AnyI)
    .Case(IAttr::getAttrName(), SpecAttrs::I)
    .Case(SIAttr::getAttrName(), SpecAttrs::SI)
    .Case(UIAttr::getAttrName(), SpecAttrs::UI)
    .Case(FAttr::getAttrName(), SpecAttrs::F)
    .Case(StringAttr::getAttrName(), SpecAttrs::String)
    .Case(TypeAttr::getAttrName(), SpecAttrs::Type)
    .Case(UnitAttr::getAttrName(), SpecAttrs::Unit)
    .Case(DictionaryAttr::getAttrName(), SpecAttrs::Dictionary)
    .Case(ElementsAttr::getAttrName(), SpecAttrs::Elements)
    .Case(ArrayAttr::getAttrName(), SpecAttrs::Array)
    .Case(SymbolRefAttr::getAttrName(), SpecAttrs::SymbolRef)
    .Case(FlatSymbolRefAttr::getAttrName(), SpecAttrs::FlatSymbolRef)
    .Case(ConstantAttr::getAttrName(), SpecAttrs::Constant)
    .Case(AnyOfAttr::getAttrName(), SpecAttrs::AnyOf)
    .Case(AllOfAttr::getAttrName(), SpecAttrs::AllOf)
    .Case(OfTypeAttr::getAttrName(), SpecAttrs::OfType)
    .Case(OptionalAttr::getAttrName(), SpecAttrs::Optional)
    .Case(DefaultAttr::getAttrName(), SpecAttrs::Default)
    .Case(IsaAttr::getAttrName(), SpecAttrs::Isa)
    .Default(SpecAttrs::NUM_ATTRS);

  if (kind == SpecAttrs::NUM_ATTRS) {
    parser.emitError(parser.getCurrentLocation(),
                     "unknown attribute constraint");
    return Attribute{};
  }
  ParseAction<Attribute, DialectAsmParser> action{parser};
  return SpecAttrs::kindSwitch(action, kind);
}

Attribute ConstantAttr::parse(DialectAsmParser &parser) {
  Attribute constAttr;
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  if (parser.parseLess() || parser.parseAttribute(constAttr) ||
      parser.parseGreater())
    return Attribute{};
  return ConstantAttr::getChecked(loc, constAttr);
}

Attribute AnyOfAttr::parse(DialectAsmParser &parser) {
  return parseAttrList<AnyOfAttr>(parser);
}

Attribute AllOfAttr::parse(DialectAsmParser &parser) {
  return parseAttrList<AllOfAttr>(parser);
}

Attribute OfTypeAttr::parse(DialectAsmParser &parser) {
  Type attrTy;
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  if (parser.parseLess() || parser.parseType(attrTy) ||
      parser.parseGreater())
    return Attribute{};
  return OfTypeAttr::getChecked(loc, attrTy);
}

Attribute OptionalAttr::parse(DialectAsmParser &parser) {
  Attribute baseAttr;
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  if (parser.parseLess() || parser.parseAttribute(baseAttr) ||
      parser.parseGreater())
    return Attribute{};
  return OptionalAttr::getChecked(loc, baseAttr);
}

Attribute DefaultAttr::parse(DialectAsmParser &parser) {
  Attribute baseAttr, defaultAttr;
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  if (parser.parseLess() || parser.parseAttribute(baseAttr) ||
      parser.parseComma() || parser.parseAttribute(defaultAttr) ||
      parser.parseGreater())
    return Attribute{};
  return DefaultAttr::getChecked(loc, baseAttr, defaultAttr);
}

Attribute IsaAttr::parse(DialectAsmParser &parser) {
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  mlir::SymbolRefAttr attrRef;
  if (parser.parseLess() || parser.parseAttribute(attrRef) ||
      parser.parseGreater())
    return Attribute{};
  return IsaAttr::getChecked(loc, attrRef);
}

} // end namespace dmc
