#include "SpecTypes.h"

namespace dmc {
namespace SpecTypes {
/// Big switch table.
template <typename ActionT>
std::result_of_t<ActionT(AnyType)>
kindSwitch(const ActionT &action, mlir::Type base) {
  assert(SpecTypes::is(base) && "Not a SpecType");
  switch (base.getKind()) {
  default:
    assert(base.isa<AnyType>() && "Unknown SpecType");
    return action.template operator()(base.cast<AnyType>());
  case None:
    return action.template operator()(base.cast<NoneType>());
  case AnyOf:
    return action.template operator()(base.cast<AnyOfType>());
  case AllOf:
    return action.template operator()(base.cast<AllOfType>());
  case AnyInteger:
    return action.template operator()(base.cast<AnyIntegerType>());
  case AnyI:
    return action.template operator()(base.cast<AnyIType>());
  case AnyIntOfWidths:
    return action.template operator()(base.cast<AnyIntOfWidthsType>());
  case AnySignlessInteger:
    return action.template operator()(base.cast<AnySignlessIntegerType>());
  case I:
    return action.template operator()(base.cast<IType>());
  case SignlessIntOfWidths:
    return action.template operator()(base.cast<SignlessIntOfWidthsType>());
  case AnySignedInteger:
    return action.template operator()(base.cast<AnySignedIntegerType>());
  case SI:
    return action.template operator()(base.cast<SIType>());
  case SignedIntOfWidths:
    return action.template operator()(base.cast<SignedIntOfWidthsType>());
  case AnyUnsignedInteger:
    return action.template operator()(base.cast<AnyUnsignedIntegerType>());
  case UI:
    return action.template operator()(base.cast<UIType>());
  case UnsignedIntOfWidths:
    return action.template operator()(base.cast<UnsignedIntOfWidthsType>());
  case Index:
    return action.template operator()(base.cast<IndexType>());
  case AnyFloat:
    return action.template operator()(base.cast<AnyFloatType>());
  case F:
    return action.template operator()(base.cast<FType>());
  case FloatOfWidths:
    return action.template operator()(base.cast<FloatOfWidthsType>());
  case BF16:
    return action.template operator()(base.cast<BF16Type>());
  case AnyComplex:
    return action.template operator()(base.cast<AnyComplexType>());
  case Complex:
    return action.template operator()(base.cast<ComplexType>());
  case Opaque:
    return action.template operator()(base.cast<OpaqueType>());
  case Variadic:
    return action.template operator()(base.cast<VariadicType>());
  case Isa:
    return action.template operator()(base.cast<IsaType>());
  }
}
} // end namespace SpecTypes
} // end namespace dmc
