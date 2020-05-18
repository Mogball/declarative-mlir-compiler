#pragma once

#include "Kinds.h"

#include <mlir/IR/DialectImplementation.h>

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

  /// Attribute hooks.
  static OpTraitAttr get(mlir::FlatSymbolRefAttr nameAttr,
                         mlir::ArrayAttr paramAttr);
  static OpTraitAttr getChecked(
      mlir::Location loc, mlir::FlatSymbolRefAttr nameAttr,
      mlir::ArrayAttr paramAttr);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, mlir::FlatSymbolRefAttr nameAttr,
      mlir::ArrayAttr paramAttr);

  /// Parsing and printing.
  static OpTraitAttr parse(mlir::DialectAsmParser &parser);
  void print(mlir::DialectAsmPrinter &printer);

  /// Getters.
  llvm::StringRef getName();
  llvm::ArrayRef<mlir::Attribute> getParameters();
};

/// An attribute representing a dynamic operation's dynamic op traits.
class OpTraitsAttr : public mlir::Attribute::AttrBase<
                     OpTraitsAttr, mlir::Attribute, detail::OpTraitsStorage> {
public:
  using Base::Base;

  static bool kindof(unsigned kind) { return kind == TraitAttr::OpTraits; }

  /// Attribute hooks.
  static OpTraitsAttr get(mlir::ArrayAttr traits);
  static OpTraitsAttr getChecked(mlir::Location loc, mlir::ArrayAttr traits);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, mlir::ArrayAttr traits);

  /// Parsing and printing.
  static OpTraitsAttr parse(mlir::DialectAsmParser &parser);
  void print(mlir::DialectAsmPrinter &printer);

  /// Getters.
  inline auto getValue() {
    return llvm::map_range(getUnderlyingValue(), [](mlir::Attribute attr)
                           { return attr.cast<OpTraitAttr>(); });
  }

private:
  llvm::ArrayRef<mlir::Attribute> getUnderlyingValue();
};

} // end namespace dmc
