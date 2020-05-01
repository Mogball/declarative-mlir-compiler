#include "dmc/Spec/SpecDialect.h"
#include "dmc/Spec/SpecOps.h"
#include "dmc/Spec/SpecTypes.h"
#include "dmc/Spec/SpecAttrs.h"

#include <mlir/IR/DialectImplementation.h>

using namespace mlir;

namespace dmc {

/// SpecDialect.
SpecDialect::SpecDialect(MLIRContext *ctx)
    : Dialect{getDialectNamespace(), ctx} {
  addOperations<
      DialectOp, DialectTerminatorOp, OperationOp
  >();
  addTypes<
      AnyType, NoneType, AnyOfType, 
      AnyIntegerType, AnyIType, AnyIntOfWidthsType,
      AnySignlessIntegerType, IType, SignlessIntOfWidthsType,
      AnySignedIntegerType, SIType, SignedIntOfWidthsType,
      AnyUnsignedIntegerType, UIType, UnsignedIntOfWidthsType,
      IndexType, AnyFloatType, FType, FloatOfWidthsType, BF16Type,
      AnyComplexType, ComplexType, OpaqueType
  >();
  addAttributes<
      AnyAttr, BoolAttr, IndexAttr, APIntAttr,
      AnyIAttr, IAttr, SIAttr, UIAttr 
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

} // end anonymous namespace

/// Type parsing.
Type SpecDialect::parseType(DialectAsmParser &parser) const {
  if (!parser.parseOptionalKeyword("Any"))
    return AnyType::get(getContext());
  if (!parser.parseOptionalKeyword("None"))
    return NoneType::get(getContext());
  if (!parser.parseOptionalKeyword("AnyOf")) 
    return AnyOfType::parse(parser);
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
  parser.emitError(parser.getCurrentLocation(), "Unknown TypeConstraint");
  return Type{};
}

Type AnyOfType::parse(DialectAsmParser &parser) {
  // any-of-type ::= `AnyOf` `<` type(`,` type)* `>`
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
  return AnyOfType::getChecked(loc, baseTypes);
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
  StringAttr dialectNameAttr, typeNameAttr;
  if (parser.parseLess() || parser.parseAttribute(dialectNameAttr) ||
      parser.parseComma() || parser.parseAttribute(typeNameAttr) ||
      parser.parseGreater())
    return Type{};
  return OpaqueType::getChecked(loc, dialectNameAttr.getValue(),
                                typeNameAttr.getValue());
}

} // end namespace dmc
