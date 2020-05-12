#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/Dynamic/DynamicDialect.h"
#include "dmc/Dynamic/DynamicOperation.h"

using namespace mlir;

namespace dmc {

DynamicDialect::DynamicDialect(StringRef name, DynamicContext *ctx)
    : Dialect{name, ctx->getContext()},
      DynamicObject{ctx} {}

DynamicOperation *DynamicDialect::createDynamicOp(StringRef name) {
  /// Allocate on heap so AbstractOperation references stay valid.
  /// Ownership must be passed to DynamicContext.
  return new DynamicOperation(name, this);
}

DynamicTypeImpl *DynamicDialect::createDynamicType(
    StringRef name, ArrayRef<Attribute> paramSpec) {
  auto *type = new DynamicTypeImpl(getDynContext(), name, paramSpec);
  getDynContext()->registerDynamicType(type);
  return type;
}

} // end namespace dmc
