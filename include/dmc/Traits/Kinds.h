#pragma once

#include "dmc/Spec/SpecKinds.h"

namespace dmc {
namespace Traits {
enum Kind {
  IsTerminator,
  IsCommutative,
  IsIsolatedFromAbove,

  OperandsAreFloatLike,
  OperandsAreSignlessIntegerLike,
  ResultsAreBoolLike,
  ResultsAreFloatLike,
  ResultsAreSignlessIntegerLike,

  SameOperandsShape,
  SameOperandsAndResultShape,
  SameOperandsElementType,
  SameOperandsAndResultElementType,
  SameOperandsAndResultType,
  SameTypeOperands,

  NOperands,
  AtLeastNOperands,
  NRegions,
  AtLeastNRegions,
  NResults,
  AtLeastNResults,
  NSuccessors,
  AtLeastNSuccessors,

  SameVariadicOperandSizes,
  SameVariadicResultSizes,
  SizedOperandSegments,
  SizedResultSegments,
  TypeConstraintTrait,
  AttrConstraintTrait,

  NUM_TRAITS
};
} // end namespace Traits

namespace TraitAttr {
enum Kinds {
  OpTrait = SpecSuccessor::LAST_SPEC_SUCCESSOR,
  OpTraits,

  LAST_TRAIT_ATTR
};
} // end namespace TraitAttr
} // end namespace dmc
