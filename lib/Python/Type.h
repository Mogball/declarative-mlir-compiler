#pragma once

#include <mlir/IR/Types.h>

namespace mlir {

std::ostream &operator<<(std::ostream &os, Type ty);

namespace py {
} // end namespace py
} // end namespace mlir
