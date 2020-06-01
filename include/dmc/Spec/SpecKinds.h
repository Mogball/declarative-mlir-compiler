#pragma once

#include "dmc/Kind.h"

namespace dmc {

namespace SpecTypes {
enum Kinds {
  Any = Kind::FIRST_SPEC_TYPE,
  None,
  AnyOf,
  AllOf,

  AnyInteger,
  AnyI,
  AnyIntOfWidths,

  AnySignlessInteger,
  I,
  SignlessIntOfWidths,

  AnySignedInteger,
  SI,
  SignedIntOfWidths,

  AnyUnsignedInteger,
  UI,
  UnsignedIntOfWidths,

  Index,

  AnyFloat,
  F,
  FloatOfWidths,
  BF16,

  AnyComplex,
  Complex,

  Opaque,
  Function,

  Variadic, // Optional is a subset of Variadic

  Isa,

  /// Generic Python type constraint.
  Py,

  LAST_SPEC_TYPE
};
} // end namespace SpecTypes

namespace SpecAttrs {
enum Kinds {
  Any = Kind::FIRST_SPEC_ATTR,
  Bool,
  Index,
  APInt,

  AnyI,
  I,
  SI,
  UI,
  F,

  String,
  Type,
  Unit,
  Dictionary,
  Elements,
  DenseElements,
  ElementsOf,
  RankedElements,
  StringElements,
  Array,
  ArrayOf,

  SymbolRef,
  FlatSymbolRef,

  Constant,
  AnyOf,
  AllOf,
  OfType,

  Optional,
  Default,

  Isa,

  /// Generic Python attribute constraint.
  Py,

  /// Non-attribute-constraint kinds.
  OpTrait,
  OpTraits,

  LAST_SPEC_ATTR
};
} // end namespace SpecAttrs

namespace SpecRegion {
enum Kinds {
  Any = SpecAttrs::LAST_SPEC_ATTR,
  Sized,
  IsolatedFromAbove,
  Variadic,

  LAST_SPEC_REGION
};
} // end namespace SpecRegion

enum TypeKinds {
  OpType = SpecTypes::LAST_SPEC_TYPE,

  LAST_KIND
};

} // end namespace dmc
