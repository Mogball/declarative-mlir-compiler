#pragma once

#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>

namespace dmc {

/// Forward declarations.
class DialectTerminatorOp;

/// Top-level Op in the SpecDialect which defines a dialect:
///
/// dmc.Dialect @MyDialect {foldHook = @myFoldHook, ...} {
///   ...
/// }
///
/// Captured in the Op region are the Dialect Operations. The attributes are
/// used to configure the generated DynamicDialect.
class DialectOp 
    : public mlir::Op<
          DialectOp, mlir::OpTrait::ZeroOperands, mlir::OpTrait::ZeroResult,
          mlir::OpTrait::IsIsolatedFromAbove, mlir::OpTrait::SymbolTable,
          mlir::OpTrait::SingleBlockImplicitTerminator<DialectTerminatorOp>::Impl,
          mlir::SymbolOpInterface::Trait> {
public:
  using Op::Op;

  static llvm::StringRef getOperationName() { return "dmc.Dialect"; }
  static void build(mlir::OpBuilder &builder, mlir::OperationState &result, 
                    llvm::StringRef name);

  /// Operation hooks.
  static mlir::ParseResult parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result);
  void print(mlir::OpAsmPrinter &printer);
  mlir::LogicalResult verify();

  /// Get the Dialect name.
  llvm::StringRef getName();

  /// Body
  mlir::Region &getBodyRegion();
  mlir::Block *getBody();
};

/// Special terminator Op for DialectOp.
class DialectTerminatorOp 
    : public mlir::Op<DialectTerminatorOp, 
                      mlir::OpTrait::ZeroOperands, mlir::OpTrait::ZeroResult,
                      mlir::OpTrait::HasParent<DialectOp>::Impl, 
                      mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  static llvm::StringRef getOperationName() { 
    return "dmc.DialectTerminator"; 
  }
  static inline void build(mlir::OpBuilder &, mlir::OperationState &) {}
};

} // end namespace dmc
