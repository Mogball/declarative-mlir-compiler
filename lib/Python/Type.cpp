#include "Support.h"

#include <mlir/IR/Types.h>

namespace mlir {

std::ostream &operator<<(std::ostream &os, Type ty) {
  return printToOs(os, ty);
}

namespace py {

} // end namespace py
} // end namespace mlir
