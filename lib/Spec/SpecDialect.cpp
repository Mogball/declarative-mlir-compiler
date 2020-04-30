#include "dmc/Spec/SpecDialect.h"
#include "dmc/Spec/SpecTypes.h"

using namespace mlir;

namespace dmc {

SpecDialect::SpecDialect(MLIRContext *ctx)
    : Dialect{getDialectNamespace(), ctx} {
  addTypes<
      AnyType, NoneType, AnyOfType, 
      AnyIntegerType, AnyIType, AnyIntOfWidthsType,
      AnySignlessIntegerType, IType, SignlessIntOfWidthsType,
      AnySignedIntegerType, SIType, SignedIntOfWidthsType,
      AnyUnsignedIntegerType, UIType, UnsignedIntOfWidthsType,
      IndexType, AnyFloatType, FType, FloatOfWidthsType, BF16Type,
      AnyComplexType, ComplexType
  >();
}

Type SpecDialect::parseType(DialectAsmParser &parser) const {
  // TODO
  return Type{};
}

void SpecDialect::printType(Type type, DialectAsmPrinter &printer) const {
  // TODO
}

} // end namespace dmc
