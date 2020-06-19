#pragma once

#include "SpecOps.h"
#include "dmc/Dynamic/DynamicContext.h"

#include <mlir/IR/Module.h>

namespace dmc {

mlir::LogicalResult registerDialect(DialectOp dialectOp, DynamicContext *ctx);
mlir::LogicalResult registerAllDialects(mlir::ModuleOp dialects,
                                        DynamicContext *ctx);

} // end namespace dmc
