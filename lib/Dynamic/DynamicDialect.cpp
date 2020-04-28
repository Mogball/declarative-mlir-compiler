#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/Dynamic/DynamicDialect.h"
#include "dmc/Dynamic/DynamicOperation.h"

using namespace mlir;

namespace dmc {

DynamicDialect::DynamicDialect(StringRef name, DynamicContext *ctx)
    : Dialect{name, ctx->getContext()},
      DynamicObject{ctx} {}

DynamicOperation *DynamicDialect::createDynamicOp(llvm::StringRef name) {
  // Allocate on heap so AbstractOperation references stay valid
  auto *op = new DynamicOperation{name, this};
  addOperation(op->getOpInfo());
  return op;
}

} // end namespace dmc
