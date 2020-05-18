#pragma once

#include "SpecAttrs.h"

namespace dmc {

namespace detail {
struct OpTraitStorage;
} // end namespace detail

class OpTraitAttr : public mlir::Attribute::AttrBase<
                    OpTraitAttr, mlir::Attribute, detail::OpTraitStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == SpecAttrs::OpTrait; }

  static OpTraitAttr get(mlir::StringAttr nameAttr, mlir::ArrayAttr paramAttr);
  static OpTraitAttr getChecked(mlir::Location loc, mlir::StringAttr nameAttr,
                            mlir::ArrayAttr paramAttr);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, mlir::StringAttr nameAttr,
      mlir::ArrayAttr paramAttr);


  llvm::StringRef getName();
  llvm::ArrayRef<mlir::Attribute> getParameters();
};

} // end namespace dmc
