#pragma once

#include <mlir/IR/MLIRContext.h>

namespace mlir {
namespace py {

/// Store a global MLIR context instance. All calls to MLIR functions through
/// the Python API will use this instance. This simplifies the Python API as
/// users will not need to pass a context handle to all function calls.
MLIRContext *getMLIRContext();

void exposeParser();

} // end namespace py
} // end namespace mlir
