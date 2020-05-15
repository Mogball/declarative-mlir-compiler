#pragma once

#include <mlir/IR/OpImplementation.h>

namespace dmc {
namespace impl {

mlir::ParseResult parseSingleAttribute(mlir::OpAsmParser &parser,
                                       mlir::Attribute &attr);

} // end namespace impl
} // end namespace dmc
