#include "dmc/Spec/SpecDialect.h"

using namespace mlir;

namespace dmc {

SpecDialect::SpecDialect(MLIRContext *ctx)
    : Dialect{getDialectNamespace(), ctx} {}

Type SpecDialect::parseType(DialectAsmParser &parser) const {
  // TODO
  return Type{}:
}

void SpecDialect::printType(Type type, DialectAsmPrinter) const {
  // TODO
}

} // end namespace dmc
