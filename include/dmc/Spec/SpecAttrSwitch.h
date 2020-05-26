#include "SpecAttrs.h"

namespace dmc {
namespace SpecAttrs {

/// Big switch table.
template <typename ActionT>
auto kindSwitch(const ActionT &action, unsigned kind) {
  switch (kind) {
  default:
    return action.template operator()<AnyAttr>();
  case Bool:
    return action.template operator()<BoolAttr>();
  case Index:
    return action.template operator()<IndexAttr>();
  case APInt:
    return action.template operator()<APIntAttr>();
  case AnyI:
    return action.template operator()<AnyIAttr>();
  case I:
    return action.template operator()<IAttr>();
  case SI:
    return action.template operator()<SIAttr>();
  case UI:
    return action.template operator()<UIAttr>();
  case F:
    return action.template operator()<FAttr>();
  case String:
    return action.template operator()<StringAttr>();
  case Type:
    return action.template operator()<TypeAttr>();
  case Unit:
    return action.template operator()<UnitAttr>();
  case Dictionary:
    return action.template operator()<DictionaryAttr>();
  case Elements:
    return action.template operator()<ElementsAttr>();
  case Array:
    return action.template operator()<ArrayAttr>();
  case SymbolRef:
    return action.template operator()<SymbolRefAttr>();
  case FlatSymbolRef:
    return action.template operator()<FlatSymbolRefAttr>();
  case Constant:
    return action.template operator()<ConstantAttr>();
  case AnyOf:
    return action.template operator()<AnyOfAttr>();
  case AllOf:
    return action.template operator()<AllOfAttr>();
  case OfType:
    return action.template operator()<OfTypeAttr>();
  case Optional:
    return action.template operator()<OptionalAttr>();
  case Default:
    return action.template operator()<DefaultAttr>();
  case Isa:
    return action.template operator()<IsaAttr>();
  }
}

template <typename ActionT>
auto kindSwitch(const ActionT &action, mlir::Attribute base) {
  assert(SpecAttrs::is(base) && "Not a SpecAttr");
  KindActionWrapper<ActionT, mlir::Attribute> wrapper{action, base};
  return kindSwitch(wrapper, base.getKind());
}

} // end namespace SpecAttrs
} // end namespace dmc
