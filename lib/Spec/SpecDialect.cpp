#include "dmc/Spec/SpecDialect.h"
#include "dmc/Spec/SpecOps.h"
#include "dmc/Spec/SpecRegionSwitch.h"
#include "dmc/Spec/SpecSuccessorSwitch.h"
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
      FunctionType, VariadicType, IsaType,

      PyType,

      OpType, OpRegion, OpSuccessor
  >();
  addAttributes<
      AnyAttr, BoolAttr, IndexAttr, APIntAttr,
      AnyIAttr, IAttr, SIAttr, UIAttr, FAttr,
      StringAttr, TypeAttr, UnitAttr,
      DictionaryAttr, ElementsAttr, DenseElementsAttr, ElementsOfAttr,
      RankedElementsAttr, StringElementsAttr, ArrayAttr, ArrayOfAttr,
      SymbolRefAttr, FlatSymbolRefAttr,
      ConstantAttr, AnyOfAttr, AllOfAttr, OfTypeAttr,
      OptionalAttr, DefaultAttr, IsaAttr,

      PyAttr,

      AnyRegion, SizedRegion, IsolatedFromAboveRegion, VariadicRegion,
      AnySuccessor, VariadicSuccessor,

      mlir::NamedParameter
  >();
}

/// printAttribute is called from printGenericOp
void SpecDialect::printAttribute(
    Attribute attr, DialectAsmPrinter &printer) const {
  if (SpecRegion::is(attr)) {
    PrintAction<llvm::raw_ostream> action{printer.getStream()};
    SpecRegion::kindSwitch(action, attr);
  } else if (SpecSuccessor::is(attr)) {
    PrintAction<llvm::raw_ostream> action{printer.getStream()};
    SpecSuccessor::kindSwitch(action, attr);
  } else if (SpecAttrs::is(attr)) {
    PrintAction<DialectAsmPrinter> action{printer};
    SpecAttrs::kindSwitch(action, attr);
  } else if (auto opRegion = attr.dyn_cast<OpRegion>()) {
    impl::printOptionalRegionList(printer, opRegion);
  } else if (auto opSucc = attr.dyn_cast<OpSuccessor>()) {
    impl::printOptionalSuccessorList(printer, opSucc);
  } else if (auto param = attr.dyn_cast<mlir::NamedParameter>()) {
    printer << "NamedParameter" << '<' << param.getName() << ": "
        << param.getConstraint() << '>';
  } else {
    llvm_unreachable("Unknown attribute kind");
  }
}

/// printType is called from printGenericOp
void SpecDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (SpecTypes::is(type)) {
    PrintAction<DialectAsmPrinter> action{printer};
    SpecTypes::kindSwitch(action, type);
  } else {
    impl::printOpType(printer, type.cast<OpType>());
  }
}

namespace {

template <typename BaseT>
Type parseTypeList(DialectAsmParser &parser) {
  // *-of-type ::= `AnyOf` `<` type(`,` type)* `>`
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  if (parser.parseLess())
    return {};
  SmallVector<Type, 4> baseTypes;
  do {
    Type baseType;
    if (parser.parseType(baseType))
      return {};
    baseTypes.push_back(baseType);
  } while (!parser.parseOptionalComma());
  if (parser.parseGreater())
    return {};
  return BaseT::getChecked(loc, baseTypes);
}

} // end anonymous namespace

/// Type parsing.
Type SpecDialect::parseType(DialectAsmParser &parser) const {
  StringRef typeName;
  if (parser.parseKeyword(&typeName))
    return {};
  auto kind = llvm::StringSwitch<unsigned>(typeName)
    .Case(AnyType::getTypeName(), AnyType::Kind)
    .Case(NoneType::getTypeName(), NoneType::Kind)
    .Case(AnyOfType::getTypeName(), AnyOfType::Kind)
    .Case(AllOfType::getTypeName(), AllOfType::Kind)
    .Case(AnyIntegerType::getTypeName(), AnyIntegerType::Kind)
    .Case(AnyIType::getTypeName(), AnyIType::Kind)
    .Case(AnyIntOfWidthsType::getTypeName(), AnyIntOfWidthsType::Kind)
    .Case(AnySignlessIntegerType::getTypeName(), AnySignlessIntegerType::Kind)
    .Case(IType::getTypeName(), IType::Kind)
    .Case(SignlessIntOfWidthsType::getTypeName(), SignlessIntOfWidthsType::Kind)
    .Case(AnySignedIntegerType::getTypeName(), AnySignedIntegerType::Kind)
    .Case(SIType::getTypeName(), SIType::Kind)
    .Case(SignedIntOfWidthsType::getTypeName(), SignedIntOfWidthsType::Kind)
    .Case(AnyUnsignedIntegerType::getTypeName(), AnyUnsignedIntegerType::Kind)
    .Case(UIType::getTypeName(), UIType::Kind)
    .Case(UnsignedIntOfWidthsType::getTypeName(), UnsignedIntOfWidthsType::Kind)
    .Case(IndexType::getTypeName(), IndexType::Kind)
    .Case(AnyFloatType::getTypeName(), AnyFloatType::Kind)
    .Case(FType::getTypeName(), FType::Kind)
    .Case(FloatOfWidthsType::getTypeName(), FloatOfWidthsType::Kind)
    .Case(BF16Type::getTypeName(), BF16Type::Kind)
    .Case(AnyComplexType::getTypeName(), AnyComplexType::Kind)
    .Case(ComplexType::getTypeName(), ComplexType::Kind)
    .Case(OpaqueType::getTypeName(), OpaqueType::Kind)
    .Case(FunctionType::getTypeName(), FunctionType::Kind)
    .Case(VariadicType::getTypeName(), VariadicType::Kind)
    .Case(IsaType::getTypeName(), IsaType::Kind)
    .Case(PyType::getTypeName(), PyType::Kind)
    .Default(SpecTypes::LAST_SPEC_TYPE);

  if (kind == SpecTypes::LAST_SPEC_TYPE) {
    parser.emitError(parser.getCurrentLocation(), "unknown type constraint");
    return {};
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
    return {};
  return ComplexType::getChecked(loc, elBaseType);
}

Type OpaqueType::parse(DialectAsmParser &parser) {
  // `Opaque` `<` string `,` string `>`
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  mlir::StringAttr dialectNameAttr, typeNameAttr;
  if (parser.parseLess() || parser.parseAttribute(dialectNameAttr) ||
      parser.parseComma() || parser.parseAttribute(typeNameAttr) ||
      parser.parseGreater())
    return {};
  return OpaqueType::getChecked(loc, dialectNameAttr.getValue(),
                                typeNameAttr.getValue());
}

Type VariadicType::parse(DialectAsmParser &parser) {
  // `Variadic` `<`type `>`
  Type baseTy;
  if (parser.parseLess() || parser.parseType(baseTy) ||
      parser.parseGreater())
    return {};
  return VariadicType::get(baseTy);
}

Type IsaType::parse(DialectAsmParser &parser) {
  // `Isa` `<` sym-ref `>`
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  mlir::SymbolRefAttr typeRef;
  if (parser.parseLess() || parser.parseAttribute(typeRef) ||
      parser.parseGreater())
    return {};
  return IsaType::getChecked(loc, typeRef);
}

Type PyType::parse(DialectAsmParser &parser) {
  // `Py` `<` `"` py-expr `"` `>`
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  mlir::StringAttr exprAttr;
  if (parser.parseLess() || parser.parseAttribute(exprAttr) ||
      parser.parseGreater())
    return {};
  return PyType::getChecked(loc, exprAttr.getValue());
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
    .Case(AnyAttr::getAttrName(), AnyAttr::Kind)
    .Case(BoolAttr::getAttrName(), BoolAttr::Kind)
    .Case(IndexAttr::getAttrName(), IndexAttr::Kind)
    .Case(APIntAttr::getAttrName(), APIntAttr::Kind)
    .Case(AnyIAttr::getAttrName(), AnyIAttr::Kind)
    .Case(IAttr::getAttrName(), IAttr::Kind)
    .Case(SIAttr::getAttrName(), SIAttr::Kind)
    .Case(UIAttr::getAttrName(), UIAttr::Kind)
    .Case(FAttr::getAttrName(), FAttr::Kind)
    .Case(StringAttr::getAttrName(), StringAttr::Kind)
    .Case(TypeAttr::getAttrName(), TypeAttr::Kind)
    .Case(UnitAttr::getAttrName(), UnitAttr::Kind)
    .Case(DictionaryAttr::getAttrName(), DictionaryAttr::Kind)
    .Case(ElementsAttr::getAttrName(), ElementsAttr::Kind)
    .Case(DenseElementsAttr::getAttrName(), DenseElementsAttr::Kind)
    .Case(ElementsOfAttr::getAttrName(), ElementsOfAttr::Kind)
    .Case(RankedElementsAttr::getAttrName(), RankedElementsAttr::Kind)
    .Case(StringElementsAttr::getAttrName(), StringElementsAttr::Kind)
    .Case(ArrayAttr::getAttrName(), ArrayAttr::Kind)
    .Case(ArrayOfAttr::getAttrName(), ArrayOfAttr::Kind)
    .Case(SymbolRefAttr::getAttrName(), SymbolRefAttr::Kind)
    .Case(FlatSymbolRefAttr::getAttrName(), FlatSymbolRefAttr::Kind)
    .Case(ConstantAttr::getAttrName(), ConstantAttr::Kind)
    .Case(AnyOfAttr::getAttrName(), AnyOfAttr::Kind)
    .Case(AllOfAttr::getAttrName(), AllOfAttr::Kind)
    .Case(OfTypeAttr::getAttrName(), OfTypeAttr::Kind)
    .Case(OptionalAttr::getAttrName(), OptionalAttr::Kind)
    .Case(DefaultAttr::getAttrName(), DefaultAttr::Kind)
    .Case(IsaAttr::getAttrName(), IsaAttr::Kind)
    .Case(PyAttr::getAttrName(), PyAttr::Kind)
    .Default(SpecAttrs::LAST_SPEC_ATTR);

  if (kind == SpecAttrs::LAST_SPEC_ATTR) {
    parser.emitError(parser.getCurrentLocation(),
                     "unknown attribute constraint");
    return Attribute{};
  }
  ParseAction<Attribute, DialectAsmParser> action{parser};
  return SpecAttrs::kindSwitch(action, kind);
}

Attribute ElementsOfAttr::parse(DialectAsmParser &parser) {
  Type elTy;
  if (parser.parseLess() || parser.parseType(elTy) ||
      parser.parseGreater())
    return Attribute{};
  return ElementsOfAttr::get(elTy);
}

Attribute RankedElementsAttr::parse(DialectAsmParser &parser) {
  SmallVector<int64_t, 3> dims;
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  if (parser.parseLess() || impl::parseIntegerList(parser, dims) ||
      parser.parseGreater())
    return {};
  return RankedElementsAttr::getChecked(loc, dims);
}

Attribute ArrayOfAttr::parse(DialectAsmParser &parser) {
  Attribute constraint;
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  if (parser.parseLess() || parser.parseAttribute(constraint) ||
      parser.parseGreater())
    return {};
  return ArrayOfAttr::getChecked(loc, constraint);
}

Attribute ConstantAttr::parse(DialectAsmParser &parser) {
  Attribute constAttr;
  if (parser.parseLess() || parser.parseAttribute(constAttr) ||
      parser.parseGreater())
    return Attribute{};
  return ConstantAttr::get(constAttr);
}

Attribute AnyOfAttr::parse(DialectAsmParser &parser) {
  return parseAttrList<AnyOfAttr>(parser);
}

Attribute AllOfAttr::parse(DialectAsmParser &parser) {
  return parseAttrList<AllOfAttr>(parser);
}

Attribute OfTypeAttr::parse(DialectAsmParser &parser) {
  Type attrTy;
  if (parser.parseLess() || parser.parseType(attrTy) ||
      parser.parseGreater())
    return Attribute{};
  return OfTypeAttr::get(attrTy);
}

Attribute OptionalAttr::parse(DialectAsmParser &parser) {
  Attribute baseAttr;
  if (parser.parseLess() || parser.parseAttribute(baseAttr) ||
      parser.parseGreater())
    return Attribute{};
  return OptionalAttr::get(baseAttr);
}

Attribute DefaultAttr::parse(DialectAsmParser &parser) {
  Attribute baseAttr, defaultAttr;
  if (parser.parseLess() || parser.parseAttribute(baseAttr) ||
      parser.parseComma() || parser.parseAttribute(defaultAttr) ||
      parser.parseGreater())
    return Attribute{};
  return DefaultAttr::get(baseAttr, defaultAttr);
}

Attribute IsaAttr::parse(DialectAsmParser &parser) {
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  mlir::SymbolRefAttr attrRef;
  if (parser.parseLess() || parser.parseAttribute(attrRef) ||
      parser.parseGreater())
    return Attribute{};
  return IsaAttr::getChecked(loc, attrRef);
}

Attribute PyAttr::parse(DialectAsmParser &parser) {
  auto loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  mlir::StringAttr exprAttr;
  if (parser.parseLess() || parser.parseAttribute(exprAttr) ||
      parser.parseGreater())
    return {};
  return PyAttr::getChecked(loc, exprAttr.getValue());
}

} // end namespace dmc
