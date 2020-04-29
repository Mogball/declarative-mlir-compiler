#include "dmc/Spec/SpecTypes.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/IR/Diagnostics.h>

#include <unordered_set>

using namespace mlir;

namespace dmc {

namespace detail {

/// An immutable list that self-sorts on creation.
template <typename T>
struct ImmutableSortedList : public llvm::SmallVector<T, 4> {
  /// Sort on creation with comparator.
  template <typename Container, typename ComparatorT>
  ImmutableSortedList(const Container &c, 
                      ComparatorT comparator = ComparatorT{}) 
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
  using KeyTy = ImmutableSortedList<unsigned>;

  explicit WidthListStorage(KeyTy key) : widths{std::move(key)} {}

  /// Compare all types.
  bool operator==(const KeyTy &key) const { return key == widths; }
  /// Hash all the widths together.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return key.hash();
  }

  /// Create the WidthListStorage.
  static WidthListStorage *construct(TypeStorageAllocator &alloc,
                                     const KeyTy &key) {
    return new (alloc.allocate<WidthListStorage>())
        WidthListStorage{key};
  }

  KeyTy widths;
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
  if (std::size(types) != std::size(tys))
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
    return emitError(loc) << "width must be one of [1, 8, 16, 32, 64]";
  }
}

LogicalResult AnyIType::verify(Type ty) const {
  return success(ty.isInteger(getImpl()->width));
}

/// AnyIntOfWidthsType implementation.
AnyIntOfWidthsType AnyIntOfWidthsType::get(MLIRContext *ctx,
                                          ArrayRef<unsigned> widths) {
  detail::ImmutableSortedList<unsigned>    
      sortedWidths{widths, std::less<unsigned>{}};
  return Base::get(ctx, SpecTypes::AnyIntOfWidths, 
                   std::move(sortedWidths));
}

AnyIntOfWidthsType AnyIntOfWidthsType::getChecked(Location loc,
                                           ArrayRef<unsigned> widths) {
  detail::ImmutableSortedList<unsigned>    
      sortedWidths{widths, std::less<unsigned>{}};
  return Base::getChecked(loc, SpecTypes::AnyIntOfWidths,
                          std::move(sortedWidths));
}

LogicalResult AnyIntOfWidthsType::verifyConstructionInvariants(
    Location loc, ArrayRef<unsigned> widths) {
  if (widths.empty()) 
    return emitError(loc) << "empty integer width list passed";
  std::unordered_set<unsigned> widthSet{std::begin(widths), 
                                        std::end(widths)}; 
  if (std::size(widthSet) != std::size(widths))
    return emitError(loc) << "repeated integer widths passed";
  for (auto width : widths) {
    if (failed(AnyIType::verifyConstructionInvariants(loc, width)))
      return failure();
  }
  return success();
}

LogicalResult AnyIntOfWidthsType::verify(Type ty) const {
  for (auto width : getImpl()->widths) {
    if (ty.isInteger(width)) 
      return success();
  }
  return failure();
}

/// IType implementation.
IType IType::get(MLIRContext *ctx, unsigned width) {
  return Base::get(ctx, SpecTypes::I, width);
}

IType IType::getChecked(Location loc, unsigned width) {
  return Base::getChecked(loc, SpecTypes::I, width);
}

LogicalResult IType::verifyConstructionInvariants(
    Location loc, unsigned width) {
  return AnyIType::verifyConstructionInvariants(loc, width);
}

LogicalResult IType::verify(Type ty) const {
  return success(ty.isSignlessInteger(getImpl()->width));
}

/// SignlessIntOfWidthsType implementation.
SignlessIntOfWidthsType SignlessIntOfWidthsType::get(
    MLIRContext *ctx, ArrayRef<unsigned> widths) {
  detail::ImmutableSortedList<unsigned>    
      sortedWidths{widths, std::less<unsigned>{}};
  return Base::get(ctx, SpecTypes::SignlessIntOfWidths, 
                   std::move(sortedWidths));
}

SignlessIntOfWidthsType SignlessIntOfWidthsType::getChecked(
    Location loc, ArrayRef<unsigned> widths) {
  detail::ImmutableSortedList<unsigned>    
      sortedWidths{widths, std::less<unsigned>{}};
  return Base::getChecked(loc, SpecTypes::SignlessIntOfWidths,
                          std::move(sortedWidths));
}

LogicalResult SignlessIntOfWidthsType::verifyConstructionInvariants(
    Location loc, ArrayRef<unsigned> widths) {
  return AnyIntOfWidthsType::verifyConstructionInvariants(loc, widths);
}

LogicalResult SignlessIntOfWidthsType::verify(Type ty) const {
  for (auto width : getImpl()->widths) {
    if (ty.isSignlessInteger(width))
      return success();
  }
  return failure();
}

/// SIType implementation.
SIType SIType::get(MLIRContext *ctx, unsigned width) {
  return Base::get(ctx, SpecTypes::SI, width);
}

SIType SIType::getChecked(Location loc, unsigned width) {
  return Base::getChecked(loc, SpecTypes::SI, width);
}

LogicalResult SIType::verifyConstructionInvariants(
    Location loc, unsigned width) {
  return AnyIType::verifyConstructionInvariants(loc, width);
}

LogicalResult SIType::verify(Type ty) const {
  return success(ty.isSignedInteger(getImpl()->width));
}

/// SignedIntOfWidthsType implementation.
SignedIntOfWidthsType SignedIntOfWidthsType::get(
    MLIRContext *ctx, ArrayRef<unsigned> widths) {
  detail::ImmutableSortedList<unsigned>
      sortedWidths{widths, std::less<unsigned>{}};
  return Base::get(ctx, SpecTypes::SignedIntOfWidths,
                   std::move(sortedWidths));
}

SignedIntOfWidthsType SignedIntOfWidthsType::getChecked(
    Location loc, ArrayRef<unsigned> widths) {
  detail::ImmutableSortedList<unsigned>
      sortedWidths{widths, std::less<unsigned>{}};
  return Base::getChecked(loc, SpecTypes::SignedIntOfWidths,
                          std::move(sortedWidths));
}

LogicalResult SignedIntOfWidthsType::verifyConstructionInvariants(
    Location loc, ArrayRef<unsigned> widths) {
  return AnyIntOfWidthsType::verifyConstructionInvariants(loc, widths);
}

LogicalResult SignedIntOfWidthsType::verify(Type ty) const {
  for (auto width : getImpl()->widths) {
    if (ty.isSignedInteger(width))
      return success();
  }
  return failure();
}

/// UIType implementation.
UIType UIType::get(MLIRContext *ctx, unsigned width) {
  return Base::get(ctx, SpecTypes::UI, width);
}

UIType UIType::getChecked(Location loc, unsigned width) {
  return Base::getChecked(loc, SpecTypes::UI, width);
}

LogicalResult UIType::verifyConstructionInvariants(
    Location loc, unsigned width) {
  return AnyIType::verifyConstructionInvariants(loc, width);
}

LogicalResult UIType::verify(Type ty) const {
  return success(ty.isUnsignedInteger(getImpl()->width));
}

/// UnsignedIntOfWidthsType implementation.
UnsignedIntOfWidthsType UnsignedIntOfWidthsType::get(
    MLIRContext *ctx, ArrayRef<unsigned> widths) {
  detail::ImmutableSortedList<unsigned>
      sortedWidths{widths, std::less<unsigned>{}};
  return Base::get(ctx, SpecTypes::UnsignedIntOfWidths,
                   std::move(sortedWidths));
}

UnsignedIntOfWidthsType UnsignedIntOfWidthsType::getChecked(
    Location loc, ArrayRef<unsigned> widths) {
  detail::ImmutableSortedList<unsigned>
      sortedWidths{widths, std::less<unsigned>{}};
  return Base::getChecked(loc, SpecTypes::UnsignedIntOfWidths,
                          std::move(sortedWidths));
}

LogicalResult UnsignedIntOfWidthsType::verifyConstructionInvariants(
    Location loc, ArrayRef<unsigned> widths) {
  return AnyIntOfWidthsType::verifyConstructionInvariants(loc, widths);
}

LogicalResult UnsignedIntOfWidthsType::verify(Type ty) const {
  for (auto width : getImpl()->widths) {
    if (ty.isUnsignedInteger(width))
      return success();
  }
  return failure();
}

} // end namespace dmc
