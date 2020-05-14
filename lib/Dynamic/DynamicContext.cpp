#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/Dynamic/DynamicDialect.h"

#include <llvm/ADT/StringMap.h>
#include <mlir/IR/Operation.h>

using namespace mlir;
using namespace llvm;

namespace dmc {

DynamicContext::DynamicContext(MLIRContext *ctx)
    : ctx{ctx},
      typeIdAlloc{getFixedTypeIDAllocator()} {}

DynamicDialect *DynamicContext::createDynamicDialect(StringRef name) {
  return new DynamicDialect{name, this};
}

} // end namespace dmc
