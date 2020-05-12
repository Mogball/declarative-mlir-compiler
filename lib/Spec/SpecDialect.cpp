#include "dmc/Spec/SpecDialect.h"
#include "dmc/Spec/SpecOps.h"
#include "dmc/Spec/SpecTypes.h"
#include "dmc/Spec/SpecAttrs.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

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
      FunctionType, VariadicType
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
Type parseSingleWidthType(DialectAsmParser &parser) {
  // *i-type ::= `*I` `<` width `>`
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  unsigned width;
  if (parser.parseLess() || parser.parseInteger(width) ||
      parser.parseGreater())
    return Type{};
  return BaseT::getChecked(loc, width);
}

template <typename BaseT>
Type parseWidthListType(DialectAsmParser &parser) {
  // *int-of-widths-type ::= `*IntOfWidths` `<` width(`,` width)* `>`
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  if (parser.parseLess())
    return Type{};
  SmallVector<unsigned, 2> widths;
  do {
    unsigned width;
    if (parser.parseInteger(width))
      return Type{};
    widths.push_back(width);
  } while (!parser.parseOptionalComma());
  if (parser.parseGreater())
    return Type{};
  return BaseT::getChecked(loc, widths);
}

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
  // TODO string switch
  if (!parser.parseOptionalKeyword("Any"))
    return AnyType::get(getContext());
  if (!parser.parseOptionalKeyword("None"))
    return NoneType::get(getContext());
  if (!parser.parseOptionalKeyword("AnyOf"))
    return parseTypeList<AnyOfType>(parser);
  if (!parser.parseOptionalKeyword("AllOf"))
    return parseTypeList<AllOfType>(parser);
  if (!parser.parseOptionalKeyword("AnyInteger"))
    return AnyIntegerType::get(getContext());
  if (!parser.parseOptionalKeyword("AnyI"))
    return parseSingleWidthType<AnyIType>(parser);
  if (!parser.parseOptionalKeyword("AnyIntOfWidths"))
    return parseWidthListType<AnyIntOfWidthsType>(parser);
  if (!parser.parseOptionalKeyword("AnySignlessInteger"))
    return AnySignlessIntegerType::get(getContext());
  if (!parser.parseOptionalKeyword("I"))
    return parseSingleWidthType<IType>(parser);
  if (!parser.parseOptionalKeyword("SignlessIntOfWidths"))
    return parseWidthListType<SignlessIntOfWidthsType>(parser);
  if (!parser.parseOptionalKeyword("AnySignedInteger"))
    return AnySignedIntegerType::get(getContext());
  if (!parser.parseOptionalKeyword("SI"))
    return parseSingleWidthType<SIType>(parser);
  if (!parser.parseOptionalKeyword("SignedIntOfWidths"))
    return parseWidthListType<SignedIntOfWidthsType>(parser);
  if (!parser.parseOptionalKeyword("AnyUnsignedInteger"))
    return AnyUnsignedIntegerType::get(getContext());
  if (!parser.parseOptionalKeyword("UI"))
    return parseSingleWidthType<UIType>(parser);
  if (!parser.parseOptionalKeyword("UnsignedIntOfWidths"))
    return parseWidthListType<UnsignedIntOfWidthsType>(parser);
  if (!parser.parseOptionalKeyword("Index"))
    return IndexType::get(getContext());
  if (!parser.parseOptionalKeyword("AnyFloat"))
    return AnyFloatType::get(getContext());
  if (!parser.parseOptionalKeyword("F"))
    return parseSingleWidthType<FType>(parser);
  if (!parser.parseOptionalKeyword("FloatOfWidths"))
    return parseWidthListType<FloatOfWidthsType>(parser);
  if (!parser.parseOptionalKeyword("BF16"))
    return BF16Type::get(getContext());
  if (!parser.parseOptionalKeyword("AnyComplex"))
    return AnyComplexType::get(getContext());
  if (!parser.parseOptionalKeyword("Complex"))
    return ComplexType::parse(parser);
  if (!parser.parseOptionalKeyword("Opaque"))
    return OpaqueType::parse(parser);
  if (!parser.parseOptionalKeyword("Function"))
    return FunctionType::get(getContext());
  if (!parser.parseOptionalKeyword("Variadic"))
    return VariadicType::parse(parser);
  parser.emitError(parser.getCurrentLocation(), "Unknown TypeConstraint");
  return Type{};
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

/// Attribute parsing.
namespace {

template <typename BaseT>
Attribute parseSizedAttr(DialectAsmParser &parser) {
  using UnderlyingT = typename BaseT::Underlying;
  return BaseT::getChecked(
      parser.getEncodedSourceLoc(parser.getCurrentLocation()),
      parseSingleWidthType<UnderlyingT>(parser).template cast<UnderlyingT>());
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
