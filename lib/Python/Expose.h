#pragma once

#include <mlir/IR/MLIRContext.h>

namespace mlir {
namespace py {

void exposeParser();
void exposeModule();
void exposeLocation();
void exposeType();

} // end namespace py
} // end namespace mlir
