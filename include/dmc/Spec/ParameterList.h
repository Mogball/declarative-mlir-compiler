#pragma once

#include "SpecKinds.h"
#include "Parsing.h"

#include <mlir/IR/Attributes.h>
#include <mlir/IR/OpDefinition.h>

namespace mlir {
namespace dmc {
namespace detail {
struct NamedParameterStorage;
} // end namespace detail

class NamedParameter : public Attribute::AttrBase<
    NamedParameter, Attribute, detail::NamedParameterStorage> {
public:
  using Base::Base;

  static NamedParameter get(StringRef name, Attribute constraint);
  static NamedParameter getChecked(Location loc, StringRef name,
                                   Attribute constraint);
  static LogicalResult verifyConstructionInvariants(
      Location loc, StringRef name, Attribute constraint);

  static bool kindof(unsigned kind) {
    return kind == ::dmc::AttrKinds::NamedParameter;
  }

  StringRef getName();
  Attribute getConstraint();
};

using NamedParameterRange = iterator_range<llvm::mapped_iterator<ArrayRef<Attribute>::iterator, NamedParameter (*)(Attribute)>>;

#include "dmc/Spec/ParameterList.h.inc"

} // end namespace dmc
} // end namespace mlir

namespace dmc {
using NamedParameterRange = mlir::dmc::NamedParameterRange;
} // end namespace dmc
