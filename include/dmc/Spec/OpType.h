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

  static OpType getChecked(
      mlir::Location loc,
      llvm::ArrayRef<llvm::StringRef> argNames,
      llvm::ArrayRef<llvm::StringRef> retNames,
      llvm::ArrayRef<mlir::Type> argTys, llvm::ArrayRef<mlir::Type> retTys);

  static bool kindof(unsigned kind) { return kind == TypeKinds::OpTypeKind; }

  unsigned getNumOperands();
  unsigned getNumResults();
  llvm::StringRef getOperandName(unsigned idx);
  llvm::StringRef getResultName(unsigned idx);
  llvm::ArrayRef<llvm::StringRef> getOperandNames();
  llvm::ArrayRef<llvm::StringRef> getResultNames();
  mlir::Type getOperandType(unsigned idx);
  mlir::Type getResultType(unsigned idx);
  llvm::ArrayRef<mlir::Type> getOperandTypes();
  llvm::ArrayRef<mlir::Type> getResultTypes();
};

} // end namespace dmc
