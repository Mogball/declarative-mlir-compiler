#include "dmc/Spec/SpecAttrs.h"

using namespace mlir;

namespace dmc {

namespace SpecAttrs {

bool is(Attribute base) {
  return Any <= base.getKind() && base.getKind() < NUM_ATTRS;
}

LogicalResult delegateVerify(Attribute base, Attribute attr) {
  assert(is(base) && "Not a SpecAttr");
  switch (base.getKind()) {
  case Any:
    return base.cast<AnyAttr>().verify(attr);
  case Bool:
    return base.cast<BoolAttr>().verify(attr);
  case Index:
    return base.cast<IndexAttr>().verify(attr);
  case APInt:
    return base.cast<APIntAttr>().verify(attr);
  case AnyI:
    return base.cast<AnyIAttr>().verify(attr);
  case I:
    return base.cast<IAttr>().verify(attr);
  case SI:
    return base.cast<SIAttr>().verify(attr);
  case UI:
    return base.cast<UIAttr>().verify(attr);
  case F:
    return base.cast<FAttr>().verify(attr);
  case String:
    return base.cast<StringAttr>().verify(attr);
  case Type:
    return base.cast<TypeAttr>().verify(attr);
  case Unit:
    return base.cast<UnitAttr>().verify(attr);
  case Dictionary:
    return base.cast<DictionaryAttr>().verify(attr);
  case Elements:
    return base.cast<ElementsAttr>().verify(attr);
  case Array:
    return base.cast<ArrayAttr>().verify(attr);
  case SymbolRef:
    return base.cast<SymbolRefAttr>().verify(attr);
  case FlatSymbolRef:
    return base.cast<FlatSymbolRefAttr>().verify(attr);
  case Constant:
    return base.cast<ConstantAttr>().verify(attr);
  case AnyOf:
    return base.cast<AnyOfAttr>().verify(attr);
  case AllOf:
    return base.cast<AllOfAttr>().verify(attr);
  case OfType:
    return base.cast<OfTypeAttr>().verify(attr);
  case Optional:
    return base.cast<OptionalAttr>().verify(attr);
  case Default:
    return base.cast<DefaultAttr>().verify(attr);
  default:
    llvm_unreachable("Unknown SpecAttr");
    return failure();
  }
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
