#pragma once

#include <mlir/IR/Types.h>

namespace mlir {
namespace dmc {
namespace detail {
struct OpTypeStorage;
};

class OpType : public Type::TypeBase<OpType, Type, detail::OpTypeStorage> {
public:
  using Base::Base;

  static OpType get(MLIRContext *ctx,
                    ArrayRef<StringRef> argNames, ArrayRef<StringRef> retNames,
                    ArrayRef<Type> argTys, ArrayRef<Type> retTys);

  static bool kindof(unsigned kind) { return kind == OpTypeKind; }
};

} // end namespace dmc
} // end namespace mlir
