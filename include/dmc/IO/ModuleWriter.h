#pragma once

#include <mlir/IR/Module.h>

#include "dmc/Dynamic/DynamicContext.h"

namespace dmc {

/// This class provides an API for writing DynamicOperations to
/// a single MLIR Module. It hides some of the gritty underworkings.
class ModuleWriter {
public:
  explicit ModuleWriter(DynamicContext *ctx)  {}
};

} // end namespace dmc
