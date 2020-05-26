#include "SpecRegion.h"
#include "Support.h"

namespace dmc {
namespace SpecRegion {

template <typename ActionT>
auto kindSwitch(const ActionT &action, unsigned kind) {
  switch (kind) {
  default:
    return action.template operator()<AnyRegion>();
  case Sized:
    return action.template operator()<SizedRegion>();
  case IsolatedFromAbove:
    return action.template operator()<IsolatedFromAboveRegion>();
  case Variadic:
    return action.template operator()<VariadicRegion>();
  }
}

template <typename ActionT>
auto kindSwitch(const ActionT &action, mlir::Attribute base) {
  assert(SpecRegion::is(base) && "Not a SpecRegion");
  KindActionWrapper<ActionT, mlir::Attribute> wrapper{action, base};
  return kindSwitch(wrapper, base.getKind());
}

} // end namespace SpecRegion
} // end namespace dmc
