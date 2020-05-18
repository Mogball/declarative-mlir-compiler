#pragma once

#include "dmc/Traits/OpTrait.h"

#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/DialectImplementation.h>

namespace dmc {
namespace impl {

/// Parse a single attribute using an OpAsmParser.
mlir::ParseResult parseSingleAttribute(mlir::OpAsmParser &parser,
                                       mlir::Attribute &attr);

/// Parse an optional parameter list.
mlir::ParseResult parseOptionalParameterList(mlir::OpAsmParser &parser,
                                             mlir::ArrayAttr &attr);
mlir::ParseResult parseOptionalParameterList(mlir::DialectAsmParser &parser,
                                             mlir::ArrayAttr &attr);

/// Parse an optional parameter list with an on-the-fly parameter modifier.
mlir::ParseResult parseOptionalParameterList(
    mlir::OpAsmParser &parser, mlir::ArrayAttr &attr,
    mlir::Attribute (modifier)(mlir::Attribute));

/// Print a parameter list.
void printOptionalParameterList(mlir::OpAsmPrinter &printer,
                                llvm::ArrayRef<mlir::Attribute> params);
void printOptionalParameterList(mlir::DialectAsmPrinter &printer,
                                llvm::ArrayRef<mlir::Attribute> params);

/// Parse and print an op trait list attribute in pretty form.
mlir::ParseResult parseOptionalOpTraitList(mlir::OpAsmParser &parser,
                                           OpTraitsAttr &traitArr);
void printOptionalOpTraitList(mlir::OpAsmPrinter &printer,
                              OpTraitsAttr traitArr);

} // end namespace impl
} // end namespace dmc
