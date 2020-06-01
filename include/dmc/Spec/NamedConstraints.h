#pragma once

#include "SpecKinds.h"

#include <mlir/IR/Attributes.h>

namespace dmc {
namespace detail {
struct NamedConstraintStorage;
} // end namespace detail

class OpRegion : public mlir::Attribute::AttrBase<
    OpRegion, mlir::Attribute, detail::NamedConstraintStorage> {
public:
  using Base::Base;

  static OpRegion getChecked(mlir::Location loc,
                             llvm::ArrayRef<llvm::StringRef> names,
                             llvm::ArrayRef<mlir::Attribute> opRegions);

  static bool kindof(unsigned kind)
  { return kind == AttrKinds::OpRegionKind; }
};

class OpSuccessor : public mlir::Attribute::AttrBase<
    OpSuccessor, mlir::Attribute, detail::NamedConstraintStorage> {
public:
  using Base::Base;

  static OpSuccessor getChecked(mlir::Location loc,
                                llvm::ArrayRef<llvm::StringRef> names,
                                llvm::ArrayRef<mlir::Attribute> opSuccs);

  static bool kindof(unsigned kind)
  { return kind == AttrKinds::OpSuccessorKind; }
};

} // end namespace dmc
