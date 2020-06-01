#pragma once

#include "SpecKinds.h"

#include <mlir/IR/Types.h>

namespace dmc {
namespace detail {
struct OpTypeStorage;
} // end namespace detail

class OpType : public mlir::Type::TypeBase<OpType, mlir::Type,
                                           detail::OpTypeStorage> {
public:
  using Base::Base;

  static OpType get(
      mlir::MLIRContext *ctx,
      llvm::ArrayRef<llvm::StringRef> argNames,
      llvm::ArrayRef<llvm::StringRef> retNames,
      llvm::ArrayRef<mlir::Type> argTys, llvm::ArrayRef<mlir::Type> retTys);

  static bool kindof(unsigned kind) { return kind == TypeKinds::OpTypeKind; }

  unsigned getNumOperands();
  unsigned getNumResults();
  llvm::StringRef getOperandName(unsigned idx);
  llvm::StringRef getResultName(unsigned idx);
  mlir::Type getOperandType(unsigned idx);
  mlir::Type getResultType(unsigned idx);
  llvm::ArrayRef<mlir::Type> getInputs();
  llvm::ArrayRef<mlir::Type> getResults();
};

} // end namespace dmc
