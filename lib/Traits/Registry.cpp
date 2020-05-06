#include "dmc/Traits/Registry.h"
#include "dmc/Traits/StandardTraits.h"
#include "dmc/Traits/SpecTraits.h"

using namespace mlir;

namespace dmc {

TraitRegistry::TraitRegistry(MLIRContext *ctx)
    : Dialect{getDialectNamespace(), ctx} {
  registerTraits<
      IsTerminator, IsCommutative, IsIsolatedFromAbove,
      OperandsAreFloatLike, OperandsAreSignlessIntegerLike,
      ResultsAreBoolLike, ResultsAreFloatLike,
      ResultsAreSignlessIntegerLike,
      SameOperandsShape, SameOperandsAndResultShape,
      SameOperandsElementType, SameOperandsAndResultElementType,
      SameOperandsAndResultType, SameTypeOperands,

      SameVariadicOperandSizes, SameVariadicResultSizes,
      SizedOperandSegments, SizedResultSegments
    >();
}

void TraitRegistry::registerTrait(StringRef name, TraitConstructor getter) {
  auto [it, inserted] = traitRegistry.try_emplace(name, getter);
  assert(inserted && "Trait has already been registered");
}

TraitRegistry::Trait TraitRegistry::lookupTrait(StringRef name) {
  auto it = traitRegistry.find(name);
  if (it == std::end(traitRegistry))
    return nullptr;
  return it->second();
}

} // end namespace dmc
