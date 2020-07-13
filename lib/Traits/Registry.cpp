#include "GenericConstructor.h"
#include "dmc/Traits/Registry.h"
#include "dmc/Traits/OpTrait.h"
#include "dmc/Traits/StandardTraits.h"
#include "dmc/Traits/SpecTraits.h"

using namespace mlir;

namespace dmc {
namespace {

/// Unwrap values stored inside attributes.
template <typename T> struct unwrap {
  auto operator()(const T &t) const { return t.getValue(); }
};

template <> struct unwrap<IntegerAttr> {
  auto operator()(IntegerAttr val) const { return val.getValue().getZExtValue(); }
};

template <> struct unwrap<Attribute> {
  auto operator()(Attribute val) const { return val; }
};

/// Get a trait constructor's signature from a function type.
template <typename> struct TraitSignature;
template <typename TraitT, typename... ArgTs>
struct TraitSignature<TraitT(ArgTs...)> {};

/// Generic trait constructor registration.
template <typename TraitT, typename... ArgTs>
void registerTrait(TraitRegistry *reg, TraitSignature<TraitT(ArgTs...)>) {
  GenericConstructor<ArgTs...> ctorObj{[](ArgTs... args) -> Trait {
    return std::make_unique<TraitT>(unwrap<ArgTs>{}(args)...);
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

template <typename TraitCtorT>
void registerTrait(TraitRegistry *reg) {
  registerTrait(reg, TraitSignature<TraitCtorT>{});
}

template <typename... TraitCtorTs>
void registerTraits(TraitRegistry *reg) {
  (void) std::initializer_list<int>{0, (registerTrait<TraitCtorTs>(reg), 0)...};
}

} // end anonymous namespace

TraitRegistry::TraitRegistry(MLIRContext *ctx)
    : Dialect{getDialectNamespace(), ctx} {
  addAttributes<
      OpTraitAttr, OpTraitsAttr
    >();
  registerTraits<
      IsTerminator(), IsCommutative(), IsIsolatedFromAbove(),
      MemoryAlloc(), MemoryFree(), MemoryRead(), MemoryWrite(),
      Alloc(Attribute), Free(Attribute), ReadFrom(Attribute), WriteTo(Attribute),
      NoSideEffects(),

      LoopLike(StringAttr, StringAttr),

      OperandsAreFloatLike(), OperandsAreSignlessIntegerLike(),
      ResultsAreBoolLike(), ResultsAreFloatLike(),
      ResultsAreSignlessIntegerLike(),

      SameOperandsShape(), SameOperandsAndResultShape(),
      SameOperandsElementType(), SameOperandsAndResultElementType(),
      SameOperandsAndResultType(), SameTypeOperands(),

      SameVariadicOperandSizes(), SameVariadicResultSizes(),
      SizedOperandSegments(), SizedResultSegments(),

      NOperands(IntegerAttr), AtLeastNOperands(IntegerAttr),
      NRegions(IntegerAttr), AtLeastNRegions(IntegerAttr),
      NResults(IntegerAttr), AtLeastNResults(IntegerAttr),
      NSuccessors(IntegerAttr), AtLeastNSuccessors(IntegerAttr),

      HasParent(StringAttr), SingleBlockImplicitTerminator(StringAttr)
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
