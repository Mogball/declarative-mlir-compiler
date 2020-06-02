#pragma once

#include "SpecKinds.h"

#include <mlir/IR/Types.h>

namespace dmc {
namespace detail {
struct OpTypeStorage;
} // end namespace detail

struct NamedType {
  llvm::StringRef name;
  mlir::Type type;
};

class OpType : public mlir::Type::TypeBase<OpType, mlir::Type,
                                           detail::OpTypeStorage> {
public:
  using Base::Base;

  static OpType getChecked(mlir::Location loc,
                           llvm::ArrayRef<NamedType> operands,
                           llvm::ArrayRef<NamedType> results);

  static bool kindof(unsigned kind) { return kind == TypeKinds::OpTypeKind; }

  llvm::ArrayRef<NamedType> getOperands();
  llvm::ArrayRef<NamedType> getResults();

  inline unsigned getNumOperands() { return std::size(getOperands()); }
  inline unsigned getNumResults() { return std::size(getResults()); }
  inline const NamedType *getOperand(unsigned idx)
  { return &getOperands()[idx]; }
  inline const NamedType *getResult(unsigned idx)
  { return &getResults()[idx]; }

  inline llvm::StringRef getOperandName(unsigned idx)
  { return getOperand(idx)->name; }
  inline llvm::StringRef getResultName(unsigned idx)
  { return getResult(idx)->name; }
  inline mlir::Type getOperandType(unsigned idx)
  { return getOperand(idx)->type; }
  inline mlir::Type getResultType(unsigned idx)
  { return getResult(idx)->type; }

  inline auto getOperandTypes()
  { return llvm::map_range(getOperands(), &unwrap); }
  inline auto getResultTypes()
  { return llvm::map_range(getResults(), &unwrap); }

  inline auto operand_begin() { return std::begin(getOperands()); }
  inline auto operand_end() { return std::end(getOperands()); }
  inline auto result_begin() { return std::begin(getResults()); }
  inline auto result_end() { return std::end(getResults()); }

private:
  static mlir::Type unwrap(const NamedType &a) { return a.type; }
};

} // end namespace dmc
