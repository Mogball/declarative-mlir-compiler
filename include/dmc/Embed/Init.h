#pragma once

namespace mlir {
class MLIRContext;
namespace py {
void init(MLIRContext *ctx);
} // end namespace py
} // end namespace mlir
