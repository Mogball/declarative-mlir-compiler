#include "dmc/Dynamic/DynamicDialect.h"
#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/Dynamic/DynamicOperation.h"

#include <mlir/IR/Operation.h>
#include <llvm/ADT/StringMap.h>

using namespace mlir;
using namespace llvm;

namespace dmc {

class DynamicDialect::Impl {
  friend class DynamicDialect;

  StringMap<std::unique_ptr<DynamicOperation>> dynOps;
  StringMap<std::unique_ptr<DynamicTypeImpl>> dynTys;
};

DynamicDialect::~DynamicDialect() = default;

DynamicDialect::DynamicDialect(StringRef name, DynamicContext *ctx)
    : Dialect{name, ctx->getContext()},
      DynamicObject{ctx},
      impl{std::make_unique<Impl>()} {}

void DynamicDialect::registerDynamicOp(DynamicOperation *op) {
  // TODO return an error if Op already exists
  impl->dynOps.try_emplace(op->getOpInfo()->name, op);
}

DynamicOperation *DynamicDialect::lookupOp(Operation *op) const {
  auto it = impl->dynOps.find(op->getName().getStringRef());
  assert(it != impl->dynOps.end() && "DynamicOperation not found");
  return it->second.get();
}

void DynamicDialect::registerDynamicType(DynamicTypeImpl *type) {
  // TODO return an error if Type already exists
  impl->dynTys.try_emplace(type->getName(), type);
}

DynamicTypeImpl *DynamicDialect::lookupType(StringRef name) const {
  auto it = impl->dynTys.find(name);
  return it == std::end(impl->dynTys) ? nullptr : it->second.get();
}


} // end namespace dmc
