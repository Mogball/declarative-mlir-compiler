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
  default:
    llvm_unreachable("Unknown SpecAttr");
    return failure();
  }
}

} // end namespace SpecAttrs
} // end namespace dmc
