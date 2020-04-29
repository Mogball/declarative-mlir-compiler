#include "dmc/Spec/SpecDialect.h"
#include "dmc/Spec/SpecTypes.h"

using namespace mlir;

namespace dmc {

SpecDialect::SpecDialect(MLIRContext *ctx)
    : Dialect{getDialectNamespace(), ctx} {
  addTypes<
      AnyType, NoneType, AnyOfType, 
      AnyIntegerType, AnyIType, AnyIntOfWidthsType
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
