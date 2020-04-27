#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/Dynamic/DynamicDialect.h"
#include "dmc/Dynamic/DynamicOperation.h"

using namespace mlir;

namespace dmc {

DynamicDialect::DynamicDialect(StringRef name, DynamicContext *ctx)
    : Dialect{name, ctx->getContext()},
      DynamicObject{ctx} {}

void DynamicDialect::registerDynamicOp(llvm::StringRef name) {
  // TODO DynamicOperation instance is tossed
  addOperation(DynamicOperation{name, this}.getOpInfo());
}

} // end namespace dmc
