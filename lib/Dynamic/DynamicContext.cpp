#include "dmc/Dynamic/DynamicContext.h"

#include <llvm/ADT/StringMap.h>
#include <mlir/IR/Operation.h>

using namespace mlir;
using namespace llvm;

namespace dmc {

class DynamicContext::Impl {
  friend class DynamicContext;

  StringMap<std::unique_ptr<DynamicOperation>> dynOps;
  StringMap<std::unique_ptr<DynamicTypeImpl>> dynTys;
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
  impl->dynOps.try_emplace(op->getOpInfo()->name, op);
}

DynamicOperation *DynamicContext::lookupOp(Operation *op) {
  auto it = impl->dynOps.find(op->getName().getStringRef());
  assert(it != impl->dynOps.end() && "DynamicOperation not found");
  return it->second.get();
}

void DynamicContext::registerDynamicType(DynamicTypeImpl *type) {
  // TODO return an error if Type already exists
  impl->dynTys.try_emplace(type->getName(), type);
}

DynamicTypeImpl *DynamicContext::lookupType(StringRef name) {
  auto it = impl->dynTys.find(name);
  assert(it != impl->dynTys.end() && "DynamicType not found");
  return it->second.get();
}

} // end namespace dmc
