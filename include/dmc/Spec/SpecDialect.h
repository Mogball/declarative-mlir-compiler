#pragma once

#include <mlir/IR/Dialect.h>

namespace dmc {

/// This dialect defines an DSL/IR that describes
/// - Dialects and their properties
/// - Operations, their types, operands, results, properties, and traits
///
/// Some properties and traits/verifiers hook into functions defined natively.
/// If generating from a higher-level DSL (e.g. Python), these may hook into
/// Python functions with MLIR bindings for complete features.
///
/// Dialect operations, types, and attributes define their own parsing and
/// printing syntax, which is used by the generated dialect.
///
/// The Spec dialect has the following (planned) Ops:
/// - A Dialect module-level Op that defines a dialect and its properties
/// - An Operation Op that defines individual operations
/// - A Type Op that defines custom types
/// - An Attribute Op that defines custom attributes
/// - Ops to define hooks into higher-level DSLs, e.g. function prototypes for
///   Python hooks
class SpecDialect : public mlir::Dialect {
public:
  explicit SpecDialect(mlir::MLIRContext *ctx);
  static llvm::StringRef getDialectNamespace() { return "dmc"; }

  /// Custom parser and printer for operand and result type specs.
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;
  void printType(mlir::Type type,
                 mlir::DialectAsmPrinter &printer) const override;
};

} // end namespace dmc
