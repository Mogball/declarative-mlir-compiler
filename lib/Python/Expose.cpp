#include "Expose.h"

using namespace mlir::py;

PYBIND11_MODULE(mlir, m) {
  exposeParser(m);
  exposeModule(m);
  exposeLocation(m);
  exposeType(m);
}
