#include "dmc/Dynamic/DynamicContext.h"
#include "dmc/Dynamic/DynamicObject.h"

namespace dmc {

DynamicObject::DynamicObject(DynamicContext *ctx)
    : typeId{ctx->getTypeIDAlloc()->allocateID()} {}

} // end namespace dmc
