#include "dmc/Spec/SpecTypes.h"

using namespace mlir;

namespace dmc {
namespace SpecTypes {

bool is(Type base) {
  return Any <= base.getKind() && base.getKind() < NUM_TYPES;
}

/// Big switch table.
LogicalResult delegateVerify(Type base, Type ty) {
  assert(is(base) && "Not a SpecType");
  switch (base.getKind()) {
  case Any:
    return base.cast<AnyType>().verify(ty);
  case None:
    return base.cast<NoneType>().verify(ty);
  case AnyOf:
    return base.cast<AnyOfType>().verify(ty);
  case AnyInteger:
    return base.cast<AnyIntegerType>().verify(ty);
  case AnyI:
    return base.cast<AnyIType>().verify(ty);
  case AnyIntOfWidths:
    return base.cast<AnyIntOfWidthsType>().verify(ty);
  case AnySignlessInteger:
    return base.cast<AnySignlessIntegerType>().verify(ty);
  case I:
    return base.cast<IType>().verify(ty);
  case SignlessIntOfWidths:
    return base.cast<SignlessIntOfWidthsType>().verify(ty);
  case AnySignedInteger:
    return base.cast<AnySignedIntegerType>().verify(ty);
  case SI:
    return base.cast<SIType>().verify(ty);
  case SignedIntOfWidths:
    return base.cast<SignedIntOfWidthsType>().verify(ty);
  case AnyUnsignedInteger:
    return base.cast<AnyUnsignedIntegerType>().verify(ty);
  case UI:
    return base.cast<UIType>().verify(ty);
  case UnsignedIntOfWidths:
    return base.cast<UnsignedIntOfWidthsType>().verify(ty);
  case Index:
    return base.cast<IndexType>().verify(ty);
  case AnyFloat:
    return base.cast<AnyFloatType>().verify(ty);
  case F:
    return base.cast<FType>().verify(ty);
  case FloatOfWidths:
    return base.cast<FloatOfWidthsType>().verify(ty);
  case BF16:
    return base.cast<BF16Type>().verify(ty);
  case AnyComplex:
    return base.cast<AnyComplexType>().verify(ty);
  case Complex:
    return base.cast<ComplexType>().verify(ty);
  default:
    llvm_unreachable("Unknown SpecType");
    return failure();
  }
}

} // end namespace SpecTypes
} // end namespace dmc
