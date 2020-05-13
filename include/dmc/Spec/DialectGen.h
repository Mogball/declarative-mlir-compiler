#pragma once

#include "dmc/Dynamic/DynamicContext.h"

#include <mlir/IR/Module.h>

namespace dmc {

mlir::LogicalResult registerAllDialects(mlir::ModuleOp dialects,
                                        DynamicContext *ctx);

} // end namespace dmc
