#include "dmc/Spec/SpecAttrSwitch.h"

using namespace mlir;

namespace dmc {

namespace SpecAttrs {

bool is(Attribute base) {
  return Any <= base.getKind() && base.getKind() < NUM_ATTRS;
}

struct VerifyAction {
  Attribute argAttr; // attribute to verify

  template <typename ConcreteType>
  LogicalResult operator()(ConcreteType base) const {
    return base.verify(argAttr);
  }
};

LogicalResult delegateVerify(Attribute base, Attribute attr) {
  VerifyAction action{attr};
  return SpecAttrs::kindSwitch(action, base);
}

} // end namespace SpecAttrs

namespace impl {

LogicalResult verifyAttribute(Operation *op, NamedAttribute attr) {
  auto opAttr = op->getAttr(attr.first);
  if (!opAttr) {
    /// A missing optional attribute is okay.
    if (attr.second.isa<OptionalAttr>())
      return success();
    return op->emitOpError("missing attribute '") << attr.first << '\'';
  }
  auto baseAttr = attr.second;
  if (SpecAttrs::is(baseAttr)) {
    if (failed(SpecAttrs::delegateVerify(baseAttr, opAttr)))
      return op->emitOpError("attribute '") << attr.first << '\''
          << ", which is '" << opAttr << "', failed to satisfy '"
          << baseAttr << '\'';
  } else if (baseAttr != opAttr)
    return op->emitOpError("expected attribute '") << attr.first << '\''
        << " to be '" << baseAttr << "' but got '" << opAttr << '\'';
  return success();
}

LogicalResult verifyAttrConstraints(
    Operation *op, mlir::DictionaryAttr opAttrs) {
  for (auto &attr : opAttrs.getValue()) {
    if (failed(verifyAttribute(op, attr)))
      return failure();
  }
  return success();
}

} // end namespace impl

} // end namespace dmc
