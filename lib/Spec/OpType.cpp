#include "dmc/Spec/OpType.h"

#include <mlir/IR/Location.h>
#include <mlir/IR/Diagnostics.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/Twine.h>

using namespace mlir;

template <typename RangeT>
bool range_equal(const RangeT &lhs, const RangeT &rhs) {
  return llvm::size(lhs) == llvm::size(rhs) &&
    std::equal(std::begin(lhs), std::end(lhs), std::begin(rhs));
}

namespace dmc {
namespace detail {

LogicalResult verifyNameTypeRanges(Location loc, ArrayRef<StringRef> names,
                                   ArrayRef<Type> types, const char *val) {
  if (llvm::size(names) != llvm::size(types))
    return emitError(loc) << "expected same number of " << val
        << " names and types";
  // Verify names are unique.
  StringSet<> nameSet;
  unsigned idx{};
  for (auto name : names) {
    auto [it, inserted] = nameSet.insert(name);
    if (!inserted)
      return emitError(loc) << "name of " << val << " #" << idx << ": '"
          << Twine{name} << "' is not unique";
    ++idx;
  }
  return success();
}

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

  LogicalResult verifyConstructionInvariants(Location loc) {
    return success(succeeded(verifyNameTypeRanges(loc, argNames, argTys,
                                                  "operand")) &&
                   succeeded(verifyNameTypeRanges(loc, retNames, retTys,
                                                  "result")));
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

OpType OpType::getChecked(
    mlir::Location loc,
    ArrayRef<StringRef> argNames, ArrayRef<StringRef> retNames,
    ArrayRef<Type> argTys, ArrayRef<Type> retTys) {
  detail::OpTypeBase opType{argNames, retNames, argTys, retTys};
  if (failed(opType.verifyConstructionInvariants(loc)))
    return {};
  return Base::get(loc.getContext(), TypeKinds::OpTypeKind, opType);
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
