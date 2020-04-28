#pragma once

#include <mlir/IR/Builders.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/Module.h>

#include "dmc/Dynamic/DynamicContext.h"

namespace dmc {

/// Forward declarations.
class DynamicOperation;

/// This class provides an API for writing DynamicOperations to
/// a single MLIR Module. It hides some of the gritty underworkings.
class ModuleWriter {
public:
  explicit ModuleWriter(DynamicContext *ctx);

  inline mlir::ModuleOp getModule() { return module; }

  mlir::FuncOp createFunction(
      llvm::StringRef name,
      llvm::ArrayRef<mlir::Type> argTys,
      llvm::ArrayRef<mlir::Type> retTys);

private:
  mlir::OpBuilder builder;
  mlir::ModuleOp module;
};

class FunctionWriter {
public:
  explicit FunctionWriter(mlir::FuncOp func);

  inline auto getArguments() { return func.getArguments(); }

  mlir::Operation *createOp(
      DynamicOperation *op, mlir::ValueRange args, 
      llvm::ArrayRef<mlir::Type> retTys);

  mlir::Operation *createOp(
      llvm::StringRef name, mlir::ValueRange args,
      llvm::ArrayRef<mlir::Type> retTys);

private:
  mlir::OpBuilder builder;
  mlir::FuncOp func;

  mlir::Block *entryBlock;

  /// Create a generic Operation.
  mlir::Operation *createOp(
      mlir::OperationName opName, mlir::ValueRange args,
      llvm::ArrayRef<mlir::Type> retTys);
};

} // end namespace dmc
