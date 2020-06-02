#include "dmc/Spec/OpType.h"
#include "dmc/Spec/NamedConstraints.h"

#include <mlir/IR/Location.h>
#include <mlir/IR/Diagnostics.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/Twine.h>

using namespace mlir;

namespace dmc {

bool operator==(const NamedType &lhs, const NamedType &rhs) {
  return lhs.name == rhs.name && lhs.type == rhs.type;
}

bool operator==(const NamedConstraint &lhs, const NamedConstraint &rhs) {
  return lhs.name == rhs.name && lhs.attr == rhs.attr;
}

llvm::hash_code hash_value(const NamedType &t) {
  return hash_combine(t.name, t.type);
}

llvm::hash_code hash_value(const NamedConstraint &t) {
  return hash_combine(t.name, t.attr);
}

template <typename NamedArgT>
static LogicalResult verifyNamedRange(Location loc, ArrayRef<NamedArgT> args,
                                      const char *val) {
  // Verify names are unique.
  StringSet<> nameSet;
  unsigned idx{};
  for (auto &arg : args) {
    auto [it, inserted] = nameSet.insert(arg.name);
    if (!inserted)
      return emitError(loc) << "name of " << val << " #" << idx << ": '"
          << Twine{arg.name} << "' is not unique";
    ++idx;
  }
  return success();
}

namespace detail {

struct OpTypeStorage : public TypeStorage {
  using KeyTy = std::pair<ArrayRef<NamedType>, ArrayRef<NamedType>>;

  explicit OpTypeStorage(ArrayRef<NamedType> operands,
                         ArrayRef<NamedType> results)
      : operands{operands}, results{results} {}

  bool operator==(const KeyTy &key) const {
    return operands == key.first && results == key.second;
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }

  static OpTypeStorage *construct(TypeStorageAllocator &alloc,
                                  const KeyTy &key) {
    return new (alloc.allocate<OpTypeStorage>())
        OpTypeStorage{alloc.copyInto(key.first), alloc.copyInto(key.second)};
  }

  ArrayRef<NamedType> operands;
  ArrayRef<NamedType> results;
};

struct NamedConstraintStorage : public AttributeStorage {
  using KeyTy = ArrayRef<NamedConstraint>;

  explicit NamedConstraintStorage(const KeyTy &key)
      : attrs{key} {}

  bool operator==(const KeyTy &key) const {
    return attrs == key;
  }

  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key);
  }

  static NamedConstraintStorage *construct(AttributeStorageAllocator &alloc,
                                           const KeyTy &key) {
    return new (alloc.allocate<NamedConstraintStorage>())
        NamedConstraintStorage{alloc.copyInto(key)};
  }

  ArrayRef<NamedConstraint> attrs;
};

} // end namespace detail

OpType OpType::getChecked(
    Location loc, ArrayRef<NamedType> operands, ArrayRef<NamedType> results) {
  if (failed(verifyNamedRange(loc, operands, "operand")) ||
      failed(verifyNamedRange(loc, results, "result")))
    return {};
  return Base::get(loc.getContext(), TypeKinds::OpTypeKind, operands, results);
}

OpRegion OpRegion::getChecked(Location loc, ArrayRef<NamedConstraint> attrs) {
  if (failed(verifyNamedRange(loc, attrs, "region")))
    return {};
  return Base::get(loc.getContext(), AttrKinds::OpRegionKind, attrs);
}

OpSuccessor OpSuccessor::getChecked(Location loc,
                                    ArrayRef<NamedConstraint> attrs) {
  if (failed(verifyNamedRange(loc, attrs, "successor")))
    return {};
  return Base::get(loc.getContext(), AttrKinds::OpSuccessorKind, attrs);
}

ArrayRef<NamedType> OpType::getOperands() { return getImpl()->operands; }
ArrayRef<NamedType> OpType::getResults() { return getImpl()->results; }
ArrayRef<NamedConstraint> OpRegion::getRegions() { return getImpl()->attrs; }
ArrayRef<NamedConstraint> OpSuccessor::getSuccessors()
{ return getImpl()->attrs; }

} // end namespace dmc
