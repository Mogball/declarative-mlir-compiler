#pragma once

#include "SpecKinds.h"

#include <mlir/IR/Attributes.h>

namespace dmc {
namespace detail {
struct NamedConstraintStorage;
} // end namespace detail

struct NamedConstraint {
  llvm::StringRef name;
  mlir::Attribute attr;
};

namespace detail {
inline mlir::Attribute unwrap(const NamedConstraint &a) { return a.attr; }
} // end namespace detail

class OpRegion : public mlir::Attribute::AttrBase<
    OpRegion, mlir::Attribute, detail::NamedConstraintStorage> {
public:
  using Base::Base;

  static OpRegion getChecked(mlir::Location loc,
                             llvm::ArrayRef<NamedConstraint> regions);

  static bool kindof(unsigned kind)
  { return kind == AttrKinds::OpRegionKind; }

  llvm::ArrayRef<NamedConstraint> getRegions();

  inline unsigned getNumRegions() { return std::size(getRegions()); }
  inline const NamedConstraint *getRegion(unsigned idx)
  { return &getRegions()[idx]; }
  inline llvm::StringRef getRegionName(unsigned idx)
  { return getRegion(idx)->name; }
  inline mlir::Attribute getRegionAttr(unsigned idx)
  { return getRegion(idx)->attr; }

  inline auto getRegionAttrs()
  { return llvm::map_range(getRegions(), detail::unwrap); }

  inline auto begin() { return std::begin(getRegionAttrs()); }
  inline auto end() { return std::end(getRegionAttrs()); }
  inline auto size() { return std::size(getRegions()); }
};

class OpSuccessor : public mlir::Attribute::AttrBase<
    OpSuccessor, mlir::Attribute, detail::NamedConstraintStorage> {
public:
  using Base::Base;

  static OpSuccessor getChecked(mlir::Location loc,
                                llvm::ArrayRef<NamedConstraint> successors);

  static bool kindof(unsigned kind)
  { return kind == AttrKinds::OpSuccessorKind; }

  llvm::ArrayRef<NamedConstraint> getSuccessors();

  inline unsigned getNumSuccessors() { return std::size(getSuccessors()); }
  inline const NamedConstraint *getSuccessor(unsigned idx)
  { return &getSuccessors()[idx]; }
  inline llvm::StringRef getSuccessorName(unsigned idx)
  { return getSuccessor(idx)->name; }
  inline mlir::Attribute getSuccessorAttr(unsigned idx)
  { return getSuccessor(idx)->attr; }

  inline auto getSuccessorAttrs()
  { return llvm::map_range(getSuccessors(), detail::unwrap); }

  inline auto begin() { return std::begin(getSuccessorAttrs()); }
  inline auto end() { return std::end(getSuccessorAttrs()); }
  inline auto size() { return std::size(getSuccessors()); }
};

} // end namespace dmc
