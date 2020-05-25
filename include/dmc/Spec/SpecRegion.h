#pragma once

#include <mlir/IR/Attributes.h>
#include <mlir/IR/Region.h>

namespace dmc {

namespace SpecRegion {
enum Kinds {
  Any,
  Sized,
  IsolatedFromAbove,
  Variadic,

  NUM_KINDS
};

bool is(mlir::Attribute base);
mlir::LogicalResult delegateVerify(mlir::Attribute base, mlir::Region *region);

} // end namespace SpecRegion

namespace detail {
struct SizedRegionAttrStorage;
struct VariadicRegionAttrStorage;
} // end namespace detail

/// Match any region.
class AnyRegion : public mlir::Attribute::AttrBase<AnyRegion> {
public:
  using Base::Base;
  static llvm::StringLiteral getName() { return "Any"; }
  static bool kindof(unsigned kind) { return kind == SpecRegion::Any; }

  static AnyRegion get(mlir::MLIRContext *ctx) {
    return Base::get(ctx, SpecRegion::Any);
  }

  inline mlir::LogicalResult verify(mlir::Region *) {
    return mlir::success();
  }
};

/// Match a region with a given number of blocks.
class SizedRegion : public mlir::Attribute::AttrBase<
    SizedRegion, mlir::Attribute, detail::SizedRegionAttrStorage> {
public:
  using Base::Base;
  static llvm::StringLiteral getName() { return "Sized"; }
  static bool kindof(unsigned kind) { return kind == SpecRegion::Sized; }

  static SizedRegion get(mlir::MLIRContext *ctx, unsigned size);
  static SizedRegion getChecked(mlir::Location loc, unsigned size);
  static mlir::LogicalResult verifyConstructionInvariants(mlir::Location loc,
                                                          unsigned size);

  mlir::LogicalResult verify(mlir::Region *region);
};

/// Match a region isolated from above.
class IsolatedFromAboveRegion : public mlir::Attribute::AttrBase<
    IsolatedFromAboveRegion> {
public:
  using Base::Base;
  static llvm::StringLiteral getName() { return "IsolatedFromAbove"; }
  static bool kindof(unsigned kind)
  { return kind == SpecRegion::IsolatedFromAbove; }

  static IsolatedFromAboveRegion get(mlir::MLIRContext *ctx) {
    return Base::get(ctx, SpecRegion::IsolatedFromAbove);
  }

  mlir::LogicalResult verify(mlir::Region *region);
};

/// Variadic regions.
class VariadicRegion : public mlir::Attribute::AttrBase<
    VariadicRegion, mlir::Attribute, detail::VariadicRegionAttrStorage> {
public:
  using Base::Base;
  static llvm::StringLiteral getName() { return "Variadic"; }
  static bool kindof(unsigned kind) { return kind == SpecRegion::Variadic; }

  static VariadicRegion get(mlir::Attribute regionConstraint);
  static VariadicRegion getChecked(mlir::Location loc,
                                   mlir::Attribute regionConstraint);
  static mlir::LogicalResult verifyConstructionInvariants(
      mlir::Location loc, mlir::Attribute regionConstraint);

  mlir::LogicalResult verify(mlir::Region *region);
};

/// Verify Region constraints.
namespace impl {
mlir::LogicalResult verifyRegionConstraints(
    mlir::Operation *op, mlir::ArrayAttr opRegions);
} // end namespace impl

} // end namespace dmc
