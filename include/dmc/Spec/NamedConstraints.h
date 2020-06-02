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

  llvm::ArrayRef<llvm::StringRef> getRegionNames();
  llvm::ArrayRef<mlir::Attribute> getRegionAttrs();
  unsigned getNumRegions();

  inline auto begin() { return std::begin(getRegionAttrs()); }
  inline auto end() { return std::end(getRegionAttrs()); }
  inline auto size() { return std::size(getRegionAttrs()); }
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

  llvm::ArrayRef<llvm::StringRef> getSuccessorNames();
  llvm::ArrayRef<mlir::Attribute> getSuccessorAttrs();
  unsigned getNumSuccessors();

  inline auto begin() { return std::begin(getSuccessorAttrs()); }
  inline auto end() { return std::end(getSuccessorAttrs()); }
  inline auto size() { return std::size(getSuccessorAttrs()); }
};

} // end namespace dmc
