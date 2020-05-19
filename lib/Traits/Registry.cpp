#include "dmc/Traits/Registry.h"
#include "dmc/Traits/OpTrait.h"
#include "dmc/Traits/StandardTraits.h"
#include "dmc/Traits/SpecTraits.h"

using namespace mlir;

namespace dmc {

namespace detail {

template <typename ArgT, unsigned I>
auto unpackArg(ArrayRef<Attribute> args) {
  return args[I].cast<ArgT>();
}

template <typename... ArgsT, typename ConstructorT, unsigned... Is>
auto callCtor(ConstructorT ctor, ArrayRef<Attribute> args,
              std::integer_sequence<unsigned, Is...>) {
  return ctor(unpackArg<ArgsT, Is>(args)...);
}

} // end namespace detail

template <typename... ArgsT>
class Constructor {
public:
  using ConstructorT = TraitRegistry::Trait (*)(ArgsT...);

  Constructor(ConstructorT ctor) : ctor(ctor) {}

  TraitRegistry::Trait operator()(ArrayRef<Attribute> args) {
    using Indices = std::make_integer_sequence<unsigned, sizeof...(ArgsT)>;
    return detail::callCtor<ArgsT...>(ctor, args, Indices{});
  }

private:
  ConstructorT ctor;
};

TraitRegistry::TraitRegistry(MLIRContext *ctx)
    : Dialect{getDialectNamespace(), ctx} {
  addAttributes<
      OpTraitAttr, OpTraitsAttr
    >();
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
