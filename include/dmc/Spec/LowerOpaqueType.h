#pragma once

#include "SpecOps.h"

namespace dmc {
mlir::LogicalResult lowerOpaqueTypes(DialectOp dialectOp);
} // end namespace dmc
