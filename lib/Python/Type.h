#pragma once

#include <mlir/IR/Types.h>

namespace mlir {
namespace py {

unsigned getIntOrFloatBitWidth(Type ty);

} // end namespace py
} // end namespace mlir
