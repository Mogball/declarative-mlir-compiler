#include "Expose.h"

#include <mlir/IR/Types.h>

using namespace mlir::py;

PYBIND11_MODULE(mlir, m) {
  exposeParser(m);
  exposeModule(m);
  auto type = exposeTypeBase(m);
  exposeAttribute(m);
  exposeType(m, type);
}
