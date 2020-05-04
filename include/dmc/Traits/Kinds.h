#pragma once

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

  NUM_TRAITS
};
} // end namespace Traits
} // end namespace dmc
