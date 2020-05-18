#pragma once

#include "SpecAttrs.h"

namespace dmc {

namespace detail {
struct OpTraitStorage;
} // end namespace detail

class OpTrait : public mlir::Attribute::AttrBase<OpTrait, mlir::Attribute,
                                                 detail::OpTraitStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == SpecAttrs::OpTrait; }

  static OpTrait get(mlir::StringAttr nameAttr, mlir::ArrayAttr paramAttr);
  static OpTrait getChecked(mlir::Location loc, mlir::StringAttr nameAttr,
                            mlir::ArrayAttr paramAttr);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, mlir::StringAttr nameAttr,
      mlir::ArrayAttr paramAttr);


  llvm::StringRef getName();
  llvm::ArrayRef<mlir::Attribute> getParameters();
};

} // end namespace dmc
