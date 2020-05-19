#include "dmc/Traits/StandardTraits.h"

using namespace mlir;

namespace dmc {

LogicalResult HasParent::verifyOp(mlir::Operation *op) const {
  auto parentOpName = op->getParentOp()->getName().getStringRef();
  if (parentOpName == parentName)
    return success();
  return op->emitOpError() << "expects parent op '" << parentName
      << "' but got '" << parentOpName << '\'';
}

} // end namespace dmc
