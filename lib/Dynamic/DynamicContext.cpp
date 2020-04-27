#include "dmc/Dynamic/DynamicContext.h"

using namespace mlir;

namespace dmc {

DynamicContext::DynamicContext(MLIRContext *ctx)
    : ctx{ctx},
      typeIdAlloc{getFixedTypeIDAllocator()} {}

DynamicDialect *DynamicContext::createDynamicDialect(StringRef name) {
  return new DynamicDialect{name, this};
}

} // end namespace dmc
