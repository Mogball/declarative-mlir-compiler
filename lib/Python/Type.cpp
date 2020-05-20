#include "Support.h"

#include <mlir/IR/Types.h>

namespace mlir {
namespace py {

unsigned getIntOrFloatBitWidth(Type ty) {
  if (!ty.isIntOrFloat())
    throw std::invalid_argument{"only integer or float types have bit widths"};
  return ty.getIntOrFloatBitWidth();
}

} // end namespace py
} // end namespace mlir
