#include "dmc/Spec/ParameterList.h"
#include "dmc/Spec/SpecAttrs.h"

using namespace dmc;

namespace mlir {
namespace dmc {
namespace impl {
LogicalResult verifyParameterList(Operation *op, ArrayRef<Attribute> params) {
  unsigned idx = 0;
  for (auto &param : params) {
    if (!SpecAttrs::is(param))
      return op->emitOpError("parameter #") << idx << " expected a SpecAttr "
          << "but got: " << param;
    ++idx;
  }
  return success();
}
} // end namespace impl
} // end namespace dmc
} // end namespace mlir
