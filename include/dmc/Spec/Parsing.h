#pragma once

#include <mlir/IR/OpImplementation.h>

namespace dmc {
namespace impl {

/// Parse a single attribute using an OpAsmParser.
mlir::ParseResult parseSingleAttribute(mlir::OpAsmParser &parser,
                                       mlir::Attribute &attr);

/// Parse an optional parameter list.
mlir::ParseResult parseOptionalParameterList(mlir::OpAsmParser &parser,
                                             mlir::ArrayAttr &attr);

/// Parse an optional parameter list with an on-the-fly parameter modifier.
mlir::ParseResult parseOptionalParameterList(
    mlir::OpAsmParser &parser, mlir::ArrayAttr &attr,
    mlir::Attribute (modifier)(mlir::Attribute));

} // end namespace impl
} // end namespace dmc
