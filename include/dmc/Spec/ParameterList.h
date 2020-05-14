#pragma once

#include <mlir/IR/OpDefinition.h>

namespace mlir {
namespace dmc {
namespace impl {
LogicalResult verifyParameterList(Operation *op, ArrayRef<Attribute> params);
void printParameterList(OpAsmPrinter &printer, ArrayRef<Attribute> params);
} // end namespace impl

#include "dmc/Spec/ParameterList.h.inc"
} // end namespace dmc
} // end namespace mlir

