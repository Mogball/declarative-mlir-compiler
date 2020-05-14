#include "SpecTypes.h"
#include "Support.h"

namespace dmc {
namespace SpecTypes {

/// Big switch table.
template <typename ActionT>
auto kindSwitch(const ActionT &action, unsigned kind) {
  switch (kind) {
  default:
    return action.template operator()<AnyType>();
  case None:
    return action.template operator()<NoneType>();
  case AnyOf:
    return action.template operator()<AnyOfType>();
  case AllOf:
    return action.template operator()<AllOfType>();
  case AnyInteger:
    return action.template operator()<AnyIntegerType>();
  case AnyI:
    return action.template operator()<AnyIType>();
  case AnyIntOfWidths:
    return action.template operator()<AnyIntOfWidthsType>();
  case AnySignlessInteger:
    return action.template operator()<AnySignlessIntegerType>();
  case I:
    return action.template operator()<IType>();
  case SignlessIntOfWidths:
    return action.template operator()<SignlessIntOfWidthsType>();
  case AnySignedInteger:
    return action.template operator()<AnySignedIntegerType>();
  case SI:
    return action.template operator()<SIType>();
  case SignedIntOfWidths:
    return action.template operator()<SignedIntOfWidthsType>();
  case AnyUnsignedInteger:
    return action.template operator()<AnyUnsignedIntegerType>();
  case UI:
    return action.template operator()<UIType>();
  case UnsignedIntOfWidths:
    return action.template operator()<UnsignedIntOfWidthsType>();
  case Index:
    return action.template operator()<IndexType>();
  case AnyFloat:
    return action.template operator()<AnyFloatType>();
  case F:
    return action.template operator()<FType>();
  case FloatOfWidths:
    return action.template operator()<FloatOfWidthsType>();
  case BF16:
    return action.template operator()<BF16Type>();
  case AnyComplex:
    return action.template operator()<AnyComplexType>();
  case Complex:
    return action.template operator()<ComplexType>();
  case Opaque:
    return action.template operator()<OpaqueType>();
  case Variadic:
    return action.template operator()<VariadicType>();
  case Isa:
    return action.template operator()<IsaType>();
  }
}

template <typename ActionT>
auto kindSwitch(const ActionT &action, mlir::Type base) {
  assert(SpecTypes::is(base) && "Not a SpecType");
  KindActionWrapper<ActionT, mlir::Type> wrapper{action, base};
  return kindSwitch(wrapper, base.getKind());
}

} // end namespace SpecTypes
} // end namespace dmc
