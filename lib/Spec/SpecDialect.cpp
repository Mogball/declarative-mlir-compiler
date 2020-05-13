#include "dmc/Spec/SpecDialect.h"
#include "dmc/Spec/SpecOps.h"
#include "dmc/Spec/SpecAttrs.h"
#include "dmc/Spec/SpecTypeSwitch.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <llvm/ADT/StringSwitch.h>

using namespace mlir;

namespace dmc {

/// SpecDialect.
SpecDialect::SpecDialect(MLIRContext *ctx)
    : Dialect{getDialectNamespace(), ctx} {
  addOperations<
      DialectOp, DialectTerminatorOp, OperationOp, TypeOp
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
      OptionalAttr, DefaultAttr
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
struct ParseAction {
  DialectAsmParser &parser;

  template <typename ConcreteType>
  Type operator()() const {
    return ConcreteType::parse(parser);
  }
};

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
  ParseAction action{parser};
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
Attribute parseSizedAttr(DialectAsmParser &parser) {
  using UnderlyingT = typename BaseT::Underlying;
  auto width = impl::parseSingleWidth(parser);
  if (!width) return {};
  return BaseT::getChecked(
      parser.getEncodedSourceLoc(parser.getCurrentLocation()),
      UnderlyingT::get(*width, parser.getBuilder().getContext()));
}

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
  assert(!type && "SpecAttr has no Type");
  /// TODO string switch
  if (!parser.parseOptionalKeyword("Any"))
    return AnyAttr::get(getContext());
  if (!parser.parseOptionalKeyword("Bool"))
    return BoolAttr::get(getContext());
  if (!parser.parseOptionalKeyword("Index"))
    return IndexAttr::get(getContext());
  if (!parser.parseOptionalKeyword("APInt"))
    return APIntAttr::get(getContext());
  if (!parser.parseOptionalKeyword("AnyI"))
    return parseSizedAttr<AnyIAttr>(parser);
  if (!parser.parseOptionalKeyword("I"))
    return parseSizedAttr<IAttr>(parser);
  if (!parser.parseOptionalKeyword("SI"))
    return parseSizedAttr<SIAttr>(parser);
  if (!parser.parseOptionalKeyword("UI"))
    return parseSizedAttr<UIAttr>(parser);
  if (!parser.parseOptionalKeyword("F"))
    return parseSizedAttr<FAttr>(parser);
  if (!parser.parseOptionalKeyword("String"))
    return StringAttr::get(getContext());
  if (!parser.parseOptionalKeyword("Type"))
    return TypeAttr::get(getContext());
  if (!parser.parseOptionalKeyword("Unit"))
    return UnitAttr::get(getContext());
  if (!parser.parseOptionalKeyword("Dictionary"))
    return DictionaryAttr::get(getContext());
  if (!parser.parseOptionalKeyword("Elements"))
    return ElementsAttr::get(getContext());
  if (!parser.parseOptionalKeyword("Array"))
    return ArrayAttr::get(getContext());
  if (!parser.parseOptionalKeyword("SymbolRef"))
    return SymbolRefAttr::get(getContext());
  if (!parser.parseOptionalKeyword("FlatSymbolRef"))
    return FlatSymbolRefAttr::get(getContext());
  if (!parser.parseOptionalKeyword("Constant"))
    return ConstantAttr::parse(parser);
  if (!parser.parseOptionalKeyword("AnyOf"))
    return parseAttrList<AnyOfAttr>(parser);
  if (!parser.parseOptionalKeyword("AllOf"))
    return parseAttrList<AllOfAttr>(parser);
  if (!parser.parseOptionalKeyword("OfType"))
    return OfTypeAttr::parse(parser);
  if (!parser.parseOptionalKeyword("Optional"))
    return OptionalAttr::parse(parser);
  if (!parser.parseOptionalKeyword("Default"))
    return DefaultAttr::parse(parser);
  parser.emitError(parser.getCurrentLocation(), "Unknown AttrConstraint");
  return Attribute{};
}

Attribute ConstantAttr::parse(DialectAsmParser &parser) {
  Attribute constAttr;
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  if (parser.parseLess() || parser.parseAttribute(constAttr) ||
      parser.parseGreater())
    return Attribute{};
  return ConstantAttr::getChecked(loc, constAttr);
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

} // end namespace dmc
