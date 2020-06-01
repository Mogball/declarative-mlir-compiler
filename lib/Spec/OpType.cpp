#include "dmc/Spec/OpType.h"
#include "dmc/Spec/NamedConstraints.h"

#include <mlir/IR/Location.h>
#include <mlir/IR/Diagnostics.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/Twine.h>

using namespace mlir;

namespace dmc {

template <typename ArgT>
LogicalResult verifyNamedRange(Location loc, ArrayRef<StringRef> names,
                               ArrayRef<ArgT> types, const char *val) {
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

namespace detail {

struct OpTypeBase {
  ArrayRef<StringRef> argNames, retNames;
  ArrayRef<Type> argTys, retTys;

  bool operator==(const OpTypeBase &o) const {
    return argNames == o.argNames && retNames == o.retNames &&
        argTys == o.argTys && retTys == o.retTys;
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
    return new (alloc.allocate<OpTypeStorage>()) OpTypeStorage{{
      alloc.copyInto(key.argNames), alloc.copyInto(key.retNames),
      alloc.copyInto(key.argTys), alloc.copyInto(key.retTys)
    }};
  }
};

struct NamedConstraintBase {
  ArrayRef<StringRef> names;
  ArrayRef<Attribute> attrs;

  bool operator==(const NamedConstraintBase &o) const {
    return names == o.names && attrs == o.attrs;
  }
};

struct NamedConstraintStorage : public AttributeStorage,
                                public NamedConstraintBase {
  using KeyTy = NamedConstraintBase;

  explicit NamedConstraintStorage(const KeyTy &key)
      : NamedConstraintBase{key} {}

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.names, key.attrs);
  }

  static NamedConstraintStorage *construct(AttributeStorageAllocator &alloc,
                                           const KeyTy &key) {
    return new (alloc.allocate<NamedConstraintStorage>())
        NamedConstraintStorage{{alloc.copyInto(key.names),
                                alloc.copyInto(key.attrs)}};
  }
};

} // end namespace detail

OpType OpType::getChecked(
    mlir::Location loc,
    ArrayRef<StringRef> argNames, ArrayRef<StringRef> retNames,
    ArrayRef<Type> argTys, ArrayRef<Type> retTys) {
  if (failed(verifyNamedRange(loc, argNames, argTys, "operand")) ||
      failed(verifyNamedRange(loc, retNames, retTys, "result")))
    return {};
  detail::OpTypeBase opType{argNames, retNames, argTys, retTys};
  return Base::get(loc.getContext(), TypeKinds::OpTypeKind, opType);
}

OpRegion OpRegion::getChecked(Location loc, ArrayRef<StringRef> names,
                              ArrayRef<Attribute> opRegions) {
  if (failed(verifyNamedRange(loc, names, opRegions, "region")))
    return {};
  return Base::get(loc.getContext(), AttrKinds::OpRegionKind,
                   detail::NamedConstraintBase{names, opRegions});
}

OpSuccessor OpSuccessor::getChecked(Location loc, ArrayRef<StringRef> names,
                                    ArrayRef<Attribute> opSuccs) {
  if (failed(verifyNamedRange(loc, names, opSuccs, "successor")))
    return {};
  return Base::get(loc.getContext(), AttrKinds::OpSuccessorKind,
                   detail::NamedConstraintBase{names, opSuccs});
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
