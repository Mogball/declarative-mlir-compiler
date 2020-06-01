#include "dmc/Spec/OpType.h"

using namespace mlir;

template <typename RangeT>
bool range_equal(const RangeT &lhs, const RangeT &rhs) {
  return llvm::size(lhs) == llvm::size(rhs) &&
    std::equal(std::begin(lhs), std::end(lhs), std::begin(rhs));
}

namespace dmc {
namespace detail {
struct OpTypeBase {
  ArrayRef<StringRef> argNames, retNames;
  ArrayRef<Type> argTys, retTys;

  bool operator==(const OpTypeBase &o) const {
    return
        range_equal(argNames, o.argNames) &&
        range_equal(retNames, o.retNames) &&
        range_equal(argTys, o.argTys) &&
        range_equal(retTys, o.retTys);
  }
};

struct OpTypeStorage : public TypeStorage, public OpTypeBase {
  using KeyTy = OpTypeBase;

  explicit OpTypeStorage(const KeyTy &key) : OpTypeBase{key} {}

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(
        hash_value(key.argNames), hash_value(key.retNames),
        hash_value(key.argTys), hash_value(key.retTys));
  }

  static OpTypeStorage *construct(TypeStorageAllocator &alloc,
                                  const KeyTy &key) {
    return new (alloc.allocate<OpTypeStorage>()) OpTypeStorage{OpTypeBase{
      alloc.copyInto(key.argNames), alloc.copyInto(key.retNames),
      alloc.copyInto(key.argTys), alloc.copyInto(key.retTys)
    }};
  }
};
} // end namespace detail

OpType OpType::get(MLIRContext *ctx,
                   ArrayRef<StringRef> argNames, ArrayRef<StringRef> retNames,
                   ArrayRef<Type> argTys, ArrayRef<Type> retTys) {
  assert(llvm::size(argNames) == llvm::size(argTys));
  assert(llvm::size(retNames) == llvm::size(retTys));
  return Base::get(ctx, TypeKinds::OpTypeKind,
                   detail::OpTypeBase{argNames, retNames, argTys, retTys});
}

unsigned OpType::getNumOperands() {
  return llvm::size(getImpl()->argTys);
}

unsigned OpType::getNumResults() {
  return llvm::size(getImpl()->retTys);
}

ArrayRef<StringRef> OpType::getOperandNames() {
  return getImpl()->argNames;
}

ArrayRef<StringRef> OpType::getResultNames() {
  return getImpl()->retNames;
}

StringRef OpType::getOperandName(unsigned idx) {
  assert(idx < getNumOperands());
  return getImpl()->argNames[idx];
}

StringRef OpType::getResultName(unsigned idx) {
  assert(idx < getNumResults());
  return getImpl()->retNames[idx];
}

ArrayRef<Type> OpType::getOperandTypes() {
  return getImpl()->argTys;
}

ArrayRef<Type> OpType::getResultTypes() {
  return getImpl()->retTys;
}

Type OpType::getOperandType(unsigned idx) {
  assert(idx < getNumOperands());
  return getImpl()->argTys[idx];
}

Type OpType::getResultType(unsigned idx) {
  assert(idx < getNumResults());
  return getImpl()->retTys[idx];
}

} // end namespace dmc
