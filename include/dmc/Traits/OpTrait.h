#pragma once

#include "Kinds.h"

namespace dmc {

namespace detail {
struct OpTraitStorage;
struct OpTraitsStorage;
} // end namespace detail

/// An attribute representing a parameterized op trait.
class OpTraitAttr : public mlir::Attribute::AttrBase<
                    OpTraitAttr, mlir::Attribute, detail::OpTraitStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == TraitAttr::OpTrait; }

  static OpTraitAttr get(mlir::StringAttr nameAttr, mlir::ArrayAttr paramAttr);
  static OpTraitAttr getChecked(mlir::Location loc, mlir::StringAttr nameAttr,
                            mlir::ArrayAttr paramAttr);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, mlir::StringAttr nameAttr,
      mlir::ArrayAttr paramAttr);


  llvm::StringRef getName();
  llvm::ArrayRef<mlir::Attribute> getParameters();
};

/// An attribute representing a dynamic operation's dynamic op traits.
class OpTraitsAttr : public mlir::Attribute::AttrBase<
                     OpTraitsAttr, mlir::Attribute, detail::OpTraitsStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == TraitAttr::OpTraits; }

  static OpTraitsAttr get(mlir::ArrayAttr traits);
  static OpTraitsAttr getChecked(mlir::Location loc, mlir::ArrayAttr traits);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, mlir::ArrayAttr traits);

  inline auto getValue() {
    return llvm::map_range(getUnderlyingValue(), [](mlir::Attribute attr)
                           { return attr.cast<OpTraitAttr>(); });
  }

private:
  llvm::ArrayRef<mlir::Attribute> getUnderlyingValue();
};

} // end namespace dmc
