#include "dmc/Dynamic/DynamicContext.h"

#include <llvm/ADT/StringMap.h>

using namespace mlir;
using namespace llvm;

namespace dmc {

class DynamicContext::Impl {
  friend class DynamicContext;

  StringMap<std::unique_ptr<DynamicOperation>> dynOps;
};

DynamicContext::~DynamicContext() = default;

DynamicContext::DynamicContext(MLIRContext *ctx)
    : impl{std::make_unique<Impl>()},
      ctx{ctx},
      typeIdAlloc{getFixedTypeIDAllocator()} {}

DynamicDialect *DynamicContext::createDynamicDialect(StringRef name) {
  return new DynamicDialect{name, this};
}

void DynamicContext::registerDynamicOp(DynamicOperation *op) {
  // TODO return an error if Op already exists
  impl->dynOps.try_emplace(op->getOpInfo().name, op);
}

} // end namespace dmc
