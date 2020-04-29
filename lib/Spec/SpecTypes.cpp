#include "dmc/Spec/SpecTypes.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/IR/Diagnostics.h>

using namespace mlir;

namespace dmc {

namespace detail {

/// An immutable list that self-sorts on creation.
template <typename T>
struct ImmutableSortedList : public llvm::SmallVector<T, 4> {
  /// Sort on creation with comparator.
  template <typename Container, typename ComparatorT>
  ImmutableSortedList(const Container &c, ComparatorT comparator) 
      : llvm::SmallVector<T, 4>{std::begin(c), std::end(c)} {
    llvm::sort(this->begin(), this->end(), comparator);
  }

  /// Compare list sizes and contents.
  bool operator==(const ImmutableSortedList<T> &other) const {
    if (this->size() != other.size())
      return false;
    return std::equal(this->begin(), this->end(), other.begin());
  }

  /// Hash list values.
  llvm::hash_code hash() const {
    return llvm::hash_combine_range(this->begin(), this->end());
  }
};

/// Comparing types by their opaque pointers will ensure consitent
/// ordering throughout a single program lifetime.
struct TypeComparator {
  bool operator()(Type lhs, Type rhs) {
    return lhs.getAsOpaquePointer() < rhs.getAsOpaquePointer();
  }
};

/// Storage for SpecTypes parameterized by a list of Types. Lists must
/// be equal regardless of element order.
struct TypeListStorage : public TypeStorage {
  /// Compound key of all the contained types.
  using KeyTy = ImmutableSortedList<Type>;

  explicit TypeListStorage(KeyTy key) : types{std::move(key)} {}

  /// Compare all types.
  bool operator==(const KeyTy &key) const { return key == types; }
  /// Hash combined all types.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return key.hash();
  }

  /// Create the TypeListStorage.
  static TypeListStorage *construct(TypeStorageAllocator &alloc,
                                    const KeyTy &key) {
    return new (alloc.allocate<TypeListStorage>())
        TypeListStorage{key};
  }

  KeyTy types;
};

/// Storage for SpecTypes parameterized by a width.
struct WidthStorage : public TypeStorage {
  /// Use width as key.
  using KeyTy = unsigned;

  explicit WidthStorage(KeyTy key) : width{key} {}

  /// Compare the width.
  bool operator==(const KeyTy &key) const { return key == width; }
  /// Hash the width.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  /// Create the WidthStorage;
  static WidthStorage *construct(TypeStorageAllocator &alloc,
      const KeyTy &key) {
    return new (alloc.allocate<WidthStorage>())
        WidthStorage{key};
  } 

  KeyTy width;
};

/// Storage for SpecTypes parameterized by a list of widths. Used
/// commonly for Integer TypeConstraints. Lists must be equal
/// regardless of element order.
struct WidthListStorage : public TypeStorage {
  /// Use list of widths as a compound key.
};

} // end namespace detail

/// AnyOfType implementation.
AnyOfType AnyOfType::get(ArrayRef<Type> tys) {
  auto *ctx = tys.front().getContext();
  detail::ImmutableSortedList<Type> types{tys, detail::TypeComparator{}};
  return Base::get(ctx, SpecTypes::AnyOf, std::move(types));
}

AnyOfType AnyOfType::getChecked(Location loc, ArrayRef<Type> tys) {
  detail::ImmutableSortedList<Type> types{tys, detail::TypeComparator{}};
  return Base::getChecked(loc, SpecTypes::AnyOf, std::move(types));
}

LogicalResult AnyOfType::verifyConstructionInvariants(
    Location loc, ArrayRef<Type> tys) {
  if (tys.empty()) 
    return emitError(loc) << "empty Type list passed to 'AnyOf'";
  llvm::SmallPtrSet<Type, 4> types{std::begin(tys), std::end(tys)};
  if (types.size() != tys.size())
    return emitError(loc) << "duplicate Types in list passed to 'AnyOf'";
  return success();
}

LogicalResult AnyOfType::verify(Type ty) const {
  // Success if the Type is found
  auto &types = getImpl()->types;
  return success(llvm::find(types, ty) != std::end(types));
}

/// AnyIType implementation.
AnyIType AnyIType::get(mlir::MLIRContext *ctx, unsigned width) {
  return Base::get(ctx, SpecTypes::AnyI, width);
}

AnyIType AnyIType::getChecked(Location loc, unsigned width) {
  return Base::getChecked(loc, SpecTypes::AnyI, width);
}

LogicalResult AnyIType::verifyConstructionInvariants(
    Location loc, unsigned width) {
  switch (width) {
  case 1:
  case 8:
  case 16:
  case 32:
  case 64:
    return success();
  default:
    return emitError(loc) << "width passed to AnyI must be one of "
                          << "[1, 8, 16, 32, 64]";
  }
}

LogicalResult AnyIType::verify(Type ty) const {
  return success(ty.isInteger(getImpl()->width));
}

} // end namespace dmc
