#include "dmc/Dynamic/DynamicDialect.h"

using namespace mlir;

namespace dmc {

DynamicDialect::DynamicDialect(StringRef name, MLIRContext *ctx)
    : Dialect(name, ctx) {
}

} // end namespace dmc
