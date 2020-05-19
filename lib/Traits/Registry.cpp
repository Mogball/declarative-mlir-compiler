#include "GenericConstructor.h"
#include "dmc/Traits/Registry.h"
#include "dmc/Traits/OpTrait.h"
#include "dmc/Traits/StandardTraits.h"
#include "dmc/Traits/SpecTraits.h"

using namespace mlir;

namespace dmc {

namespace {

template <typename TraitT>
void registerStatelessTrait(TraitRegistry *reg) {
  GenericConstructor<> ctorObj{[]() -> Trait {
    return std::make_unique<TraitT>();
  }};
  TraitConstructor traitCtor{
    [ctorObj](Location loc, ArrayRef<Attribute> args) {
      return ctorObj.verifySignature(loc, args);
    },
    [ctorObj](ArrayRef<Attribute> args) {
      return ctorObj.callConstructor(args);
    }
  };
  reg->registerTrait(TraitT::getName(), std::move(traitCtor));
}

template <typename... TraitTs>
void registerStatelessTraits(TraitRegistry *reg) {
  (void) std::initializer_list<int>{0,
      (registerStatelessTrait<TraitTs>(reg), 0)...};
}

} // end anonymous namespace

TraitRegistry::TraitRegistry(MLIRContext *ctx)
    : Dialect{getDialectNamespace(), ctx} {
  addAttributes<
      OpTraitAttr, OpTraitsAttr
    >();
  registerStatelessTraits<
      IsTerminator, IsCommutative, IsIsolatedFromAbove,
      OperandsAreFloatLike, OperandsAreSignlessIntegerLike,
      ResultsAreBoolLike, ResultsAreFloatLike,
      ResultsAreSignlessIntegerLike,
      SameOperandsShape, SameOperandsAndResultShape,
      SameOperandsElementType, SameOperandsAndResultElementType,
      SameOperandsAndResultType, SameTypeOperands,

      SameVariadicOperandSizes, SameVariadicResultSizes,
      SizedOperandSegments, SizedResultSegments
    >(this);
}

void TraitRegistry::registerTrait(StringRef name, TraitConstructor &&getter) {
  auto [it, inserted] = traitRegistry.try_emplace(
      name, std::forward<TraitConstructor>(getter));
  assert(inserted && "Trait has already been registered");
}

TraitConstructor TraitRegistry::lookupTrait(StringRef name) {
  auto it = traitRegistry.find(name);
  if (it == std::end(traitRegistry))
    return nullptr;
  return it->second;
}

} // end namespace dmc
