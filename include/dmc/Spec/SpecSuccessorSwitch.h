#include "SpecSuccessor.h"
#include "Support.h"

namespace dmc {
namespace SpecSuccessor {

template <typename ActionT>
auto kindSwitch(const ActionT &action, unsigned kind) {
  switch (kind) {
  default:
    return action.template operator()<AnySuccessor>();
  case Variadic:
    return action.template operator()<VariadicSuccessor>();
  }
}

template <typename ActionT>
auto kindSwitch(const ActionT &action, mlir::Attribute base) {
  assert(SpecSuccessor::is(base) && "Not a SpecSuccessor");
  KindActionWrapper<ActionT, mlir::Attribute> wrapper{action, base};
  return kindSwitch(wrapper, base.getKind());
}

} // end namespace SpecSuccessor
} // end namespace dmc
